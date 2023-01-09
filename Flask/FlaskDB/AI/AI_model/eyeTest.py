def overlay(image, x, y, w, h, overlay_image): # 대상 이미지 (3채널), x, y 좌표, width, height, 덮어씌울 이미지 (4채널:투명도를 가짐)
    alpha = overlay_image[:, :, 3]   # BGR
    image_alpha = alpha/ 255  # 0 ~ 255 -> 255 로 나누면 0 ~ 1 사이의 값 (1: 불투명, 0: 완전투명)
    for c in range(3): # channel BGR
        image[y-h:y+h, x-w:x+w, c] = (overlay_image[:, :, c] * image_alpha) + (image[y-h:y+h, x-w:x+w, c] * (1 - image_alpha))
