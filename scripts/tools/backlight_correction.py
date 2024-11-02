import cv2
import numpy as np


class BacklightCorrection:
    def __init__(self):
        pass
    
    def _backlight_correction(self, input_cvimg: np.ndarray) -> np.ndarray:
        # Convert the input image to grayscale
        gray_image = cv2.cvtColor(input_cvimg, cv2.COLOR_BGR2GRAY)

        # Gamma correction for brightness adjustment
        # gamma = 1.2
        gamma = 1.0
        # gamma = 2.0
        # gamma = 2.2
        corrected_gamma = np.power(gray_image / 255.0, 1.0 / gamma) * 255
        corrected_gamma = corrected_gamma.astype(np.uint8)

        # Multi-scale CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement
        # clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
        clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(64, 64))
        corrected1 = clahe1.apply(corrected_gamma)
        corrected2 = clahe2.apply(corrected_gamma)
        corrected_gray = cv2.addWeighted(corrected1, 0.5, corrected2, 0.5, 0)

        # Convert back to YUV to retain color information
        yuv_image = cv2.cvtColor(input_cvimg, cv2.COLOR_BGR2YUV)
        yuv_image[:, :, 0] = corrected_gray # Replace Y channel with corrected gray
        corrected_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

        return corrected_image

    def _detect_backlight(self, input_cvimg: np.ndarray) -> bool:
        # Convert the input image to grayscale for analysis
        gray_image = cv2.cvtColor(input_cvimg, cv2.COLOR_BGR2GRAY)

        # Detect highlight areas (very bright regions)
        _, highlight_thresh = cv2.threshold(
            gray_image, 200, 255, cv2.THRESH_BINARY
        )

        # Detect shadow areas (very dark regions)
        _, shadow_thresh = cv2.threshold(
            gray_image, 50, 255, cv2.THRESH_BINARY_INV
        )

        # Extract the central region of the image for analysis
        h, w = input_cvimg.shape[:2]
        center_x, center_y = w // 2, h // 2
        roi_size = min(h, w) // 4
        roi = gray_image[
            center_y - roi_size : center_y + roi_size,
            center_x - roi_size : center_x + roi_size,
        ]

        # Count the number of highlight and shadow pixels within the ROI
        num_highlight_pixels_roi = np.count_nonzero(
            highlight_thresh[
                center_y - roi_size : center_y + roi_size,
                center_x - roi_size : center_x + roi_size,
            ]
        )
        num_shadow_pixels_roi = np.count_nonzero(
            shadow_thresh[
                center_y - roi_size : center_y + roi_size,
                center_x - roi_size : center_x + roi_size,
            ]
        )

        # Calculate the histogram of the grayscale image
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        highlight_hist_peak = np.sum(hist[200:])
        shadow_hist_peak = np.sum(hist[:50])

        # Total number of pixels in the image
        total_pixels = input_cvimg.shape[0] * input_cvimg.shape[1]

        # Calculate the ratio of highlight and shadow pixels within the ROI
        highlight_ratio = num_highlight_pixels_roi / (
            roi_size * 2 * roi_size * 2
        )
        shadow_ratio = num_shadow_pixels_roi / (roi_size * 2 * roi_size * 2)

        # Calculate the histogram ratio from the total histogram
        highlight_hist_ratio = highlight_hist_peak / total_pixels
        shadow_hist_ratio = shadow_hist_peak / total_pixels

        # Calculate contrast within the ROI
        contrast = np.std(roi)

        # Adjust the conditions for backlight detection based on the calculated ratios and contrast
        if (
            (highlight_ratio > 0.1 and shadow_ratio > 0.1) or contrast < 20
        ) or (highlight_hist_ratio > 0.1 and shadow_hist_ratio > 0.1):
            return True # Backlight detected
        else:
            return False # No backlight detected

    def _preprocess(self, _input_cvimg: np.ndarray, param) -> np.ndarray:
        # Check for backlight presence and apply correction if necessary
        if self._detect_backlight(_input_cvimg) and param.do_preprocess:
            return self._backlight_correction(_input_cvimg) # Apply correction
        else:
            return _input_cvimg
