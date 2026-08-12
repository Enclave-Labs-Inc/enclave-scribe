from scribe.postprocess.parser import clean_output, remove_det_markers


def test_remove_det_strips_markers():
    raw = "<|det|>text [0,0,100,50]<|/det|>Hello world"
    result = remove_det_markers(raw)
    assert "Hello world" in result
    assert "<|det|>" not in result


def test_remove_det_drops_image_blocks():
    raw = "<|det|>text [0,0,50,50]<|/det|>Some text\n<|det|>image [0,50,100,100]<|/det|>caption"
    result = remove_det_markers(raw)
    assert "Some text" in result
    assert "caption" not in result


def test_clean_output_collapses_blank_lines():
    result = clean_output("line1\n\n\n\nline2")
    assert "\n\n\n" not in result
    assert "line1" in result and "line2" in result
