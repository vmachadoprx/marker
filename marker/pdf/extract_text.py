import os
from typing import List, Optional, Dict

import pypdfium2 as pdfium
import pypdfium2.internal as pdfium_i

from marker.pdf.utils import font_flags_decomposer
from marker.settings import settings
from marker.schema.block import Span, Line, Block
from marker.schema.page import Page
#from pdftext.extraction import dictionary_output

os.environ["TESSDATA_PREFIX"] = settings.TESSDATA_PREFIX

from functools import partial
from typing import List
from concurrent.futures import ProcessPoolExecutor
import math
import pypdfium2 as pdfium

from pdftext.inference import inference
from pdftext.model import get_model
from pdftext.pdf.chars import get_pdfium_chars
from pdftext.pdf.utils import unnormalize_bbox
from pdftext.postprocessing import merge_text, sort_blocks, postprocess_text, handle_hyphens
from pdftext.settings import settings as settings_pdftext


def _get_page_range(pdf_bytes_stream, model, page_range):
    pdf_doc = pdfium.PdfDocument(pdf_bytes_stream)
    text_chars = get_pdfium_chars(pdf_doc, page_range)
    pages = inference(text_chars, model)
    return pages


def _get_pages(pdf_bytes_stream, model=None, page_range=None, workers=None):
    if model is None:
        model = get_model()
    pdf_doc = pdfium.PdfDocument(pdf_bytes_stream)
    if page_range is None:
        page_range = range(len(pdf_doc))

    if workers is not None:
        workers = min(workers, len(page_range) // settings_pdftext.WORKER_PAGE_THRESHOLD) # It's inefficient to have too many workers, since we batch in inference
    #workers = 16
    if workers is None or workers <= 1:
        text_chars = get_pdfium_chars(pdf_doc, page_range)
        return inference(text_chars, model)

    func = partial(_get_page_range, pdf_bytes_stream, model)
    page_range = list(page_range)

    pages_per_worker = math.ceil(len(page_range) / workers)
    page_range_chunks = [page_range[i * pages_per_worker:(i + 1) * pages_per_worker] for i in range(workers)]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pages = list(executor.map(func, page_range_chunks))
    ordered_pages = [page for sublist in pages for page in sublist]

    return ordered_pages


def plain_text_output(pdf_path, sort=False, model=None, hyphens=False, page_range=None, workers=None) -> str:
    text = paginated_plain_text_output(pdf_path, sort=sort, model=model, hyphens=hyphens, page_range=page_range, workers=workers)
    return "\n".join(text)


def paginated_plain_text_output(pdf_path, sort=False, model=None, hyphens=False, page_range=None, workers=None) -> List[str]:
    pages = _get_pages(pdf_path, model, page_range, workers=workers)
    text = []
    for page in pages:
        text.append(merge_text(page, sort=sort, hyphens=hyphens).strip())
    return text


def _process_span(span, page_width, page_height, keep_chars):
    span["bbox"] = unnormalize_bbox(span["bbox"], page_width, page_height)
    span["text"] = handle_hyphens(postprocess_text(span["text"]), keep_hyphens=True)
    if not keep_chars:
        del span["chars"]
    else:
        for char in span["chars"]:
            char["bbox"] = unnormalize_bbox(char["bbox"], page_width, page_height)


def dictionary_output(pdf_bytes_stream, sort=False, model=None, page_range=None, keep_chars=False, workers=None):
    pages = _get_pages(pdf_bytes_stream, model, page_range, workers=workers)
    for page in pages:
        page_width, page_height = page["width"], page["height"]
        for block in page["blocks"]:
            for k in list(block.keys()):
                if k not in ["lines", "bbox"]:
                    del block[k]
            block["bbox"] = unnormalize_bbox(block["bbox"], page_width, page_height)
            for line in block["lines"]:
                for k in list(line.keys()):
                    if k not in ["spans", "bbox"]:
                        del line[k]
                line["bbox"] = unnormalize_bbox(line["bbox"], page_width, page_height)
                for span in line["spans"]:
                    _process_span(span, page_width, page_height, keep_chars)

        if sort:
            page["blocks"] = sort_blocks(page["blocks"])

    return pages


def pdftext_format_to_blocks(page, pnum: int) -> Page:
    page_blocks = []
    span_id = 0
    for block_idx, block in enumerate(page["blocks"]):
        block_lines = []
        for l in block["lines"]:
            spans = []
            for i, s in enumerate(l["spans"]):
                block_text = s["text"]
                # Remove trailing newlines and carriage returns (tesseract)
                while len(block_text) > 0 and block_text[-1] in ["\n", "\r"]:
                    block_text = block_text[:-1]

                block_text = block_text.replace("-\n", "") # Remove hyphenated line breaks
                span_obj = Span(
                    text=block_text, # Remove end of line newlines, not spaces
                    bbox=s["bbox"],
                    span_id=f"{pnum}_{span_id}",
                    font=f"{s['font']['name']}_{font_flags_decomposer(s['font']['flags'])}", # Add font flags to end of font
                    font_weight=s["font"]["weight"],
                    font_size=s["font"]["size"],
                )
                spans.append(span_obj)  # Text, bounding box, span id
                span_id += 1
            line_obj = Line(
                spans=spans,
                bbox=l["bbox"],
            )
            # Only select valid lines, with positive bboxes
            if line_obj.area >= 0:
                block_lines.append(line_obj)
        block_obj = Block(
            lines=block_lines,
            bbox=block["bbox"],
            pnum=pnum
        )
        # Only select blocks with lines
        if len(block_lines) > 0:
            page_blocks.append(block_obj)

    page_bbox = page["bbox"]
    page_width = abs(page_bbox[2] - page_bbox[0])
    page_height = abs(page_bbox[3] - page_bbox[1])
    rotation = page["rotation"]

    # Flip width and height if rotated
    if rotation == 90 or rotation == 270:
        page_width, page_height = page_height, page_width

    char_blocks = page["blocks"]
    page_bbox = [0, 0, page_width, page_height]
    out_page = Page(
        blocks=page_blocks,
        pnum=page["page"],
        bbox=page_bbox,
        rotation=rotation,
        char_blocks=char_blocks
    )
    return out_page


def get_text_blocks(pdf_bytes_stream, fname, max_pages: Optional[int] = None, start_page: Optional[int] = None) -> (List[Page], Dict):
    doc = pdfium.PdfDocument(pdf_bytes_stream)
    toc = get_toc(doc)

    if start_page:
        assert start_page < len(doc)
    else:
        start_page = 0

    if max_pages:
        if max_pages + start_page > len(doc):
            max_pages = len(doc) - start_page
    else:
        max_pages = len(doc) - start_page

    page_range = range(start_page, start_page + max_pages)

    char_blocks = dictionary_output(pdf_bytes_stream, page_range=page_range, keep_chars=True, workers=settings.PDFTEXT_CPU_WORKERS)
    marker_blocks = [pdftext_format_to_blocks(page, pnum) for pnum, page in enumerate(char_blocks)]

    return marker_blocks, toc


def naive_get_text(doc):
    full_text = ""
    for page_idx in range(len(doc)):
        page = doc.get_page(page_idx)
        text_page = page.get_textpage()
        full_text += text_page.get_text_bounded() + "\n"
    return full_text


def get_toc(doc, max_depth=15):
    toc = doc.get_toc(max_depth=max_depth)
    toc_list = []
    for item in toc:
        list_item = {
            "title": item.title,
            "level": item.level,
            "is_closed": item.is_closed,
            "n_kids": item.n_kids,
            "page_index": item.page_index,
            "view_mode": pdfium_i.ViewmodeToStr.get(item.view_mode),
            "view_pos": item.view_pos,
        }
        toc_list.append(list_item)
    return toc_list


def get_length_of_text(fname: str) -> int:
    doc = pdfium.PdfDocument(fname)
    text = naive_get_text(doc).strip()

    return len(text)
