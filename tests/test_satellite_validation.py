from PIL import Image, ImageDraw

from utils.preprocessing import validate_uploaded_image


def test_validate_uploaded_image_rejects_non_satellite_upload():
    invalid_image = Image.new("RGB", (256, 256), color=(255, 0, 0))

    is_valid, image_np, message = validate_uploaded_image(invalid_image)

    assert not is_valid
    assert image_np is None
    assert message == "Invalid image. Please upload a satellite image."


def test_validate_uploaded_image_rejects_portrait_like_image():
    portrait_img = Image.new("RGB", (512, 512), color=(120, 100, 90))
    draw = ImageDraw.Draw(portrait_img)
    # Draw a large central face-like circle and body shape
    draw.ellipse((156, 96, 356, 296), fill=(220, 180, 160))
    draw.ellipse((216, 146, 256, 186), fill=(50, 40, 30))
    draw.ellipse((276, 146, 316, 186), fill=(50, 40, 30))
    draw.rectangle((216, 246, 316, 346), fill=(220, 180, 160))

    is_valid, image_np, message = validate_uploaded_image(portrait_img)

    assert not is_valid
    assert image_np is None
    assert message == "Invalid image. Please upload a satellite image."
