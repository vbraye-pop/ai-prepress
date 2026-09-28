from ai_prepress.face_detect import square_crop_around


def test_square_crop_around_is_square_and_centered():
    box = square_crop_around((100, 100, 150, 150), image_size=(1000, 1000), margin=2.0, top_bias=0.0)
    x0, y0, x1, y1 = box
    assert x1 - x0 == y1 - y0
    assert x1 - x0 == 100  # margin * size = 2.0 * 50


def test_square_crop_around_shifts_instead_of_shrinking_near_edges():
    # a box in the corner would go negative if not shifted
    box = square_crop_around((0, 0, 40, 40), image_size=(200, 200), margin=3.0, top_bias=0.0)
    x0, y0, x1, y1 = box
    assert x0 >= 0 and y0 >= 0
    assert x1 - x0 == y1 - y0 == 120  # still full size, just shifted onto the canvas


def test_square_crop_around_clamps_when_crop_exceeds_image():
    box = square_crop_around((10, 10, 20, 20), image_size=(30, 30), margin=5.0, top_bias=0.0)
    x0, y0, x1, y1 = box
    assert x0 == 0 and y0 == 0
    assert x1 == 30 and y1 == 30
