from unittest.mock import MagicMock

from PIL import Image

from scribe.infer.local import infer_image


def test_infer_image_cleans_output(tmp_path):
    img = tmp_path / "test.png"
    Image.new("RGB", (64, 64)).save(img)

    mock_model = MagicMock()
    mock_model.infer.return_value = "<|det|>text [0,0,10,10]<|/det|>parsed text"

    result = infer_image(mock_model, str(img))
    assert "parsed text" in result
    assert "<|det|>" not in result
    mock_model.infer.assert_called_once()
