"""
Chart & Figure Analyzer — Multimodal Document Intelligence
==========================================================

Extracts charts/graphs/figures from PDF pages, classifies them,
and converts visual data into structured text representations
that can be embedded alongside textual chunks for cross-modal
contradiction detection.

Architecture
------------
1. PAGE RENDERING  — Render each PDF page as a high-res image (PyMuPDF)
2. IMAGE EXTRACTION — Pull embedded images via xref + render full pages
3. SMART FILTERING  — Heuristic + LLM-based filtering of logos/decorative imgs
4. VISION ANALYSIS  — Groq Llama-3.2-Vision extracts structured chart data
5. TEXT SYNTHESIS    — Convert structured data into embeddable text chunks

Why page rendering instead of just xref images?
------------------------------------------------
Many PDFs encode charts as *vector graphics* (not raster images), so
`page.get_images()` returns nothing.  Rendering the full page as a
high-resolution raster and sending it to a vision model catches BOTH
raster and vector charts — a critical robustness improvement.

References
----------
- ChartCheck (ACL 2024)  — explainable fact-checking on chart images
- ChartQAPro (2025)      — harder benchmark for complex real-world charts
- M3D (EMNLP 2024)       — multimodal inconsistency detection
"""

import os
import io
import re
import json
import time
import base64
import hashlib
import logging
from typing import Optional
from dataclasses import dataclass, field, asdict

import fitz  # PyMuPDF
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChartData:
    """Structured representation of an extracted chart/figure."""
    chart_id:       str
    page_number:    int
    chart_type:     str                         # bar, line, pie, table, scatter, other
    title:          str         = ""
    x_axis_label:   str         = ""
    y_axis_label:   str         = ""
    data_points:    list        = field(default_factory=list)   # [{label: str, value: str/float}, ...]
    trends:         str         = ""            # natural-language trend summary
    key_findings:   str         = ""            # most important takeaway
    raw_description: str        = ""            # full vision model description
    image_bytes:    bytes       = field(default=b"", repr=False)
    confidence:     float       = 0.0           # model's self-assessed confidence

    def to_text_chunk(self) -> str:
        """
        Synthesise a rich textual representation suitable for embedding
        in the same vector space as document text chunks.

        The representation is designed so that BM25 keyword matching AND
        dense semantic similarity will surface it when the text discusses
        related numbers, trends, or categories.
        """
        parts = [f"[CHART on Page {self.page_number}]"]

        if self.title:
            parts.append(f"Chart Title: {self.title}")
        parts.append(f"Chart Type: {self.chart_type}")

        if self.x_axis_label or self.y_axis_label:
            axes = []
            if self.x_axis_label:
                axes.append(f"X-axis: {self.x_axis_label}")
            if self.y_axis_label:
                axes.append(f"Y-axis: {self.y_axis_label}")
            parts.append("Axes: " + ", ".join(axes))

        if self.data_points:
            dp_strs = []
            for dp in self.data_points[:15]:        # cap for embedding length
                if isinstance(dp, dict):
                    dp_strs.append(
                        f"{dp.get('label', '?')}: {dp.get('value', '?')}"
                    )
                else:
                    dp_strs.append(str(dp))
            parts.append("Data Points: " + "; ".join(dp_strs))

        if self.trends:
            parts.append(f"Trend: {self.trends}")

        if self.key_findings:
            parts.append(f"Key Finding: {self.key_findings}")

        return "\n".join(parts)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("image_bytes", None)          # don't serialise raw bytes
        return d


# ─────────────────────────────────────────────────────────────────────────────
# VISION MODEL PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

CHART_CLASSIFICATION_PROMPT = """Look at this image from a PDF document page.

Determine if this image contains a DATA CHART or GRAPH (bar chart, line graph,
pie chart, scatter plot, area chart, histogram, table with numerical data,
infographic with statistics, or any other data visualization).

Respond with EXACTLY one of:
- CHART: <type> (e.g., CHART: bar, CHART: line, CHART: pie, CHART: table, CHART: scatter, CHART: area, CHART: infographic)
- NOT_CHART: <reason> (e.g., NOT_CHART: company logo, NOT_CHART: decorative image, NOT_CHART: photograph)

Respond with ONLY the classification line, nothing else."""


CHART_EXTRACTION_PROMPT = """You are a precise data extraction system. Analyze this chart/graph image
and extract ALL visible data into a structured JSON format.

You MUST respond with ONLY valid JSON — no markdown fences, no explanation.

Required JSON structure:
{
    "chart_type": "<bar|line|pie|scatter|area|table|histogram|infographic|other>",
    "title": "<chart title if visible, else empty string>",
    "x_axis_label": "<x-axis label if visible, else empty string>",
    "y_axis_label": "<y-axis label if visible, else empty string>",
    "data_points": [
        {"label": "<category or x-value>", "value": "<y-value or percentage>"},
        ...
    ],
    "trends": "<1-2 sentence description of the overall trend or pattern>",
    "key_findings": "<single most important quantitative takeaway>",
    "raw_description": "<3-5 sentence detailed description of what the chart shows, including colors, comparisons, and any notable features>",
    "confidence": <float 0.0-1.0, your confidence in the accuracy of the extracted data>
}

IMPORTANT RULES:
1. Extract ALL visible data points, not just a sample.
2. For percentages, include the % symbol in the value.
3. For monetary values, include currency symbols.
4. If exact values aren't readable, give your best estimate and lower confidence.
5. Describe trends precisely: "increased from X to Y" not just "increased".
6. Note any anomalies, outliers, or notable patterns in key_findings."""


PAGE_CHART_DETECTION_PROMPT = """Look at this PDF page carefully.

Does this page contain ANY data visualizations such as:
- Bar charts, line graphs, pie charts, scatter plots
- Area charts, histograms, waterfall charts
- Data tables with numerical values
- Infographics with statistics or percentages
- Any other form of quantitative visual representation

Respond with EXACTLY one line:
- CHARTS_FOUND: <count> (e.g., CHARTS_FOUND: 2)
- NO_CHARTS

Respond with ONLY the classification line."""


# ─────────────────────────────────────────────────────────────────────────────
# CORE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class ChartAnalyzer:
    """
    End-to-end chart extraction and analysis engine.

    Uses a two-phase approach:
      Phase 1 — Render full PDF pages and detect which pages have charts.
      Phase 2 — For pages with charts, run detailed structured extraction.

    This avoids wasting expensive vision API calls on text-only pages.
    """

    # Vision model on Groq — Llama 3.2 90B-Vision for best accuracy
    # Falls back to 11B if 90B is unavailable
    VISION_MODELS = [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "llama-3.2-90b-vision-preview",
        "llama-3.2-11b-vision-preview",
    ]

    # Minimum page dimensions to even consider (skip tiny thumbnails)
    MIN_PAGE_DIM = 200  # pixels

    # DPI for page rendering (higher = more accurate but slower)
    RENDER_DPI = 200

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY required for chart analysis.")
        self.client = Groq(api_key=self.api_key)
        self._working_model = None      # resolved on first call

    # ── Public API ────────────────────────────────────────────────────────

    def analyze_pdf(
        self,
        file_bytes: bytes,
        progress_callback=None,
        max_charts: int = 30,
        max_scan_pages: int = 60,
        api_delay: float = 2.0,
    ) -> list[ChartData]:
        """
        Full pipeline: PDF bytes → list of ChartData objects.

        Args:
            file_bytes:        raw PDF content
            progress_callback: fn(current_page, total_pages, status_msg)
            max_charts:        safety cap on extracted charts
            max_scan_pages:    max pages to scan in Phase 1 (saves tokens)
            api_delay:         seconds to wait between vision API calls

        Returns:
            list of ChartData with structured data + embeddable text
        """
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = len(doc)
        scan_pages = min(total_pages, max_scan_pages)
        charts: list[ChartData] = []

        # Phase 1 — detect which pages contain charts
        pages_with_charts: list[int] = []

        for page_num in range(scan_pages):
            if progress_callback:
                progress_callback(
                    page_num, scan_pages,
                    f"Scanning page {page_num + 1}/{scan_pages} for charts…"
                )

            page = doc[page_num]
            page_img = self._render_page(page)
            if page_img is None:
                continue

            # Rate-limit between Phase 1 calls
            if page_num > 0:
                time.sleep(api_delay)

            try:
                has_charts = self._detect_charts_on_page(page_img)
            except RuntimeError:
                logger.warning(f"Stopping Phase 1 scan at page {page_num + 1} (rate limit)")
                break

            if has_charts:
                pages_with_charts.append(page_num)

        if not pages_with_charts:
            doc.close()
            return []

        # Phase 2 — detailed extraction on chart pages
        for idx, page_num in enumerate(pages_with_charts):
            if len(charts) >= max_charts:
                break

            if progress_callback:
                progress_callback(
                    idx, len(pages_with_charts),
                    f"Extracting chart data from page {page_num + 1}…"
                )

            page = doc[page_num]
            page_img = self._render_page(page)
            if page_img is None:
                continue

            # Rate-limit between Phase 2 calls
            if idx > 0:
                time.sleep(api_delay)

            try:
                extracted = self._extract_chart_data(page_img, page_num)
            except RuntimeError:
                logger.warning(f"Stopping Phase 2 at page {page_num + 1} (rate limit)")
                break

            if extracted:
                charts.append(extracted)

            # Also try extracting individual embedded images
            embedded = self._extract_embedded_images(doc, page, page_num)
            for img_bytes in embedded:
                if len(charts) >= max_charts:
                    break
                time.sleep(api_delay)
                try:
                    chart_type = self._classify_image(img_bytes)
                except RuntimeError:
                    break
                if chart_type:
                    time.sleep(api_delay)
                    try:
                        extracted_emb = self._extract_chart_data(
                            img_bytes, page_num, is_embedded=True
                        )
                    except RuntimeError:
                        break
                    if extracted_emb:
                        # Deduplicate against full-page extraction
                        if not self._is_duplicate(extracted_emb, charts):
                            charts.append(extracted_emb)

        doc.close()
        return charts

    def charts_to_chunks(self, charts: list[ChartData]) -> list[dict]:
        """
        Convert ChartData objects into chunk dicts compatible with the
        existing pipeline (same schema as text chunks from document_processor).

        Chart chunks get IDs starting at 10001 to avoid collision with
        text chunk IDs (which start at 1).
        """
        chunks = []
        for i, chart in enumerate(charts):
            text = chart.to_text_chunk()
            chart_id = 10001 + i
            label = f"Chart-{i + 1} (p.{chart.page_number + 1})"
            display = f"Chart-{i + 1}: {chart.title[:35]}" if chart.title else label
            heading = chart.title or f"Chart on page {chart.page_number + 1}"

            chunks.append({
                "id":           chart_id,
                "label":        label,
                "display":      display,
                "heading":      heading,
                "text":         text,
                "is_chart":     True,
                "chart_type":   chart.chart_type,
                "page_number":  chart.page_number + 1,
                "chart_data":   chart.to_dict(),
            })
        return chunks

    # ── Internal Methods ──────────────────────────────────────────────────

    def _render_page(self, page: fitz.Page) -> Optional[bytes]:
        """Render a PDF page to PNG bytes at configured DPI."""
        try:
            mat = fitz.Matrix(self.RENDER_DPI / 72, self.RENDER_DPI / 72)
            pix = page.get_pixmap(matrix=mat)

            if pix.width < self.MIN_PAGE_DIM or pix.height < self.MIN_PAGE_DIM:
                return None

            return pix.tobytes("png")
        except Exception as e:
            logger.warning(f"Page render failed: {e}")
            return None

    def _extract_embedded_images(
        self, doc: fitz.Document, page: fitz.Page, page_num: int
    ) -> list[bytes]:
        """
        Extract embedded raster images from a PDF page.
        Applies size + uniqueness filtering to skip logos/icons.
        """
        images = []
        seen_hashes = set()

        try:
            image_list = page.get_images(full=True)
        except Exception:
            return images

        for img_info in image_list:
            xref = img_info[0]
            try:
                pix = fitz.Pixmap(doc, xref)

                # Size filter: skip tiny images (logos, icons, bullets)
                if pix.width < 150 or pix.height < 150:
                    continue

                # Aspect ratio filter: skip extremely narrow images (dividers)
                aspect = max(pix.width, pix.height) / max(1, min(pix.width, pix.height))
                if aspect > 8:
                    continue

                # Convert CMYK to RGB if needed
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                img_bytes = pix.tobytes("png")

                # Dedup by content hash
                h = hashlib.md5(img_bytes).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                images.append(img_bytes)

            except Exception as e:
                logger.debug(f"Image extraction failed for xref {xref}: {e}")
                continue

        return images

    def _get_vision_model(self) -> str:
        """Resolve the best available vision model on Groq."""
        if self._working_model:
            return self._working_model

        for model in self.VISION_MODELS:
            try:
                # Quick probe with a tiny image
                self.client.chat.completions.create(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Say OK"},
                        ]
                    }],
                    max_tokens=5,
                )
                self._working_model = model
                logger.info(f"Using vision model: {model}")
                return model
            except Exception:
                continue

        raise RuntimeError(
            "No vision model available on Groq. "
            "Ensure your API key has access to Llama Vision models."
        )

    def _call_vision(
        self,
        image_bytes: bytes,
        prompt: str,
        max_tokens: int = 800,
        retries: int = 4,
    ) -> str:
        """
        Send an image + prompt to the Groq vision model with
        exponential backoff on rate-limit (429) errors.
        """
        model = self._get_vision_model()
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64}"
                                }
                            },
                        ]
                    }],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                err = str(e).lower()
                if "rate_limit" in err or "429" in err or "too many" in err:
                    wait = 2 ** attempt * 10          # 10s, 20s, 40s, 80s
                    logger.warning(
                        f"Rate limited (attempt {attempt + 1}/{retries}), "
                        f"waiting {wait}s…"
                    )
                    time.sleep(wait)
                else:
                    logger.warning(f"Vision call failed: {e}")
                    raise

        raise RuntimeError("Vision API: max retries exceeded due to rate limits.")

    def _detect_charts_on_page(self, page_img: bytes) -> bool:
        """Phase 1: Quick check — does this page contain any charts?"""
        try:
            result = self._call_vision(page_img, PAGE_CHART_DETECTION_PROMPT, max_tokens=30)
            return result.startswith("CHARTS_FOUND")
        except Exception as e:
            logger.warning(f"Chart detection failed: {e}")
            return False

    def _classify_image(self, img_bytes: bytes) -> Optional[str]:
        """Classify an embedded image as chart type or None."""
        try:
            result = self._call_vision(img_bytes, CHART_CLASSIFICATION_PROMPT, max_tokens=30)
            if result.startswith("CHART:"):
                return result.split(":", 1)[1].strip().lower()
            return None
        except Exception:
            return None

    # Fallback prompt for when the full extraction prompt returns bad JSON
    _SIMPLE_EXTRACTION_PROMPT = (
        "Describe this chart in valid JSON with these keys ONLY: "
        '{"chart_type":"","title":"","trends":"","key_findings":"","raw_description":"","confidence":0.5}\n'
        "Respond with ONLY JSON, no other text."
    )

    def _extract_chart_data(
        self,
        img_bytes: bytes,
        page_num: int,
        is_embedded: bool = False,
    ) -> Optional[ChartData]:
        """Phase 2: Detailed structured extraction from a chart image.
        Falls back to a simpler prompt if the first attempt returns bad JSON."""
        result = None
        for attempt, prompt in enumerate([CHART_EXTRACTION_PROMPT, self._SIMPLE_EXTRACTION_PROMPT]):
            try:
                max_tok = 1200 if attempt == 0 else 400
                result = self._call_vision(img_bytes, prompt, max_tokens=max_tok)

                # Clean markdown fences if present
                result = re.sub(r'^```[a-z]*\n?', '', result, flags=re.MULTILINE)
                result = re.sub(r'```$', '', result, flags=re.MULTILINE)
                result = result.strip()

                data = json.loads(result)

                chart_id = hashlib.md5(
                    f"p{page_num}_{data.get('title', '')}_{is_embedded}".encode()
                ).hexdigest()[:10]

                return ChartData(
                    chart_id=chart_id,
                    page_number=page_num,
                    chart_type=data.get("chart_type", "other"),
                    title=data.get("title", ""),
                    x_axis_label=data.get("x_axis_label", ""),
                    y_axis_label=data.get("y_axis_label", ""),
                    data_points=data.get("data_points", []),
                    trends=data.get("trends", ""),
                    key_findings=data.get("key_findings", ""),
                    raw_description=data.get("raw_description", ""),
                    image_bytes=img_bytes,
                    confidence=float(data.get("confidence", 0.5)),
                )

            except json.JSONDecodeError as e:
                if attempt == 0:
                    logger.info(f"Chart JSON parse failed on page {page_num}, retrying with simpler prompt…")
                    time.sleep(2)
                    continue
                logger.warning(f"Chart JSON parse failed on page {page_num} (both attempts): {e}")
                return None

        return None

    def _is_duplicate(self, candidate: ChartData, existing: list[ChartData]) -> bool:
        """
        Heuristic deduplication: if a chart on the same page has a very
        similar title and same type, it's likely the same chart extracted
        from both the full-page render and an embedded image.
        """
        for ex in existing:
            if ex.page_number != candidate.page_number:
                continue
            if ex.chart_type != candidate.chart_type:
                continue
            # Title similarity (simple check)
            if (ex.title and candidate.title and
                    ex.title.lower().strip() == candidate.title.lower().strip()):
                return True
            # Data point overlap
            if ex.data_points and candidate.data_points:
                ex_labels = {str(dp.get("label", "")).lower() for dp in ex.data_points if isinstance(dp, dict)}
                cd_labels = {str(dp.get("label", "")).lower() for dp in candidate.data_points if isinstance(dp, dict)}
                if ex_labels and cd_labels:
                    overlap = len(ex_labels & cd_labels) / max(1, len(ex_labels | cd_labels))
                    if overlap > 0.7:
                        return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE — standalone usage
# ─────────────────────────────────────────────────────────────────────────────

def analyze_pdf_charts(file_bytes: bytes, progress_callback=None) -> list[ChartData]:
    """One-call convenience wrapper."""
    analyzer = ChartAnalyzer()
    return analyzer.analyze_pdf(file_bytes, progress_callback=progress_callback)


def charts_to_text_chunks(charts: list[ChartData]) -> list[dict]:
    """Convert extracted charts into pipeline-compatible chunk dicts."""
    analyzer = ChartAnalyzer.__new__(ChartAnalyzer)
    return analyzer.charts_to_chunks(charts)
