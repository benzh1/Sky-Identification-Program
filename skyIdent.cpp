#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    cv::VideoCapture capture("movie.avi");
    // If the video cannot be opened print a suitable error message to standard error
    if (!capture.isOpened()) {
        std::cerr << "Could not open video file" << std::endl;
        return -1;
    }

    while (true) {
        cv::Mat frame;
        // If the frame cannot be read then exit out of the loop
        if (!capture.read(frame)) {
            break;
        }

        // Converting the image to 320p
        cv::Mat bgr_img;
        cv::resize(frame, bgr_img, cv::Size(320, 240));

        // Converting from BGR to HSV format
        cv::Mat img;
        cv::cvtColor(bgr_img, img, cv::COLOR_BGR2HSV);

        // Converting from BGR to greyscale
        cv::Mat img_grey;
        cv::cvtColor(bgr_img, img_grey, cv::COLOR_BGR2GRAY);

        // Storing the height and width of the frame
        int h = bgr_img.rows;
        int w = bgr_img.cols;

        // Colour masks
        // Declare each of the masks in a single line
        cv::Mat blue_mask, grey_mask;
        // Using the inRange function to keep only the pixels in the image matching the specified constraints for each mask
        cv::inRange(img, cv::Scalar(95, 25, 120), cv::Scalar(130, 255, 255), blue_mask);
        cv::inRange(img, cv::Scalar(0, 0, 150), cv::Scalar(180, 70, 255), grey_mask);

        // Creating an empty colour mask for the image
        cv::Mat colour_mask = cv::Mat::zeros(h, w, CV_8UC1);

        cv::bitwise_or(blue_mask, grey_mask, colour_mask);

        // Laplacian texture calculations
        cv::Mat laplacian;
        cv::Laplacian(img_grey, laplacian, CV_32F, 3);

        // Finding the absolute textural values to ignore the direciton
        cv::Mat abs_texture;
        abs_texture = cv::abs(laplacian);

        // Smoothing the textural value to reduce noise in the image
        cv::Mat smooth_texture;
        cv::GaussianBlur(abs_texture, smooth_texture, cv::Size(5, 5), 0);

        // Normalising the texture to be between 1 and 0 so that it can be used as a probability
        double max_val;
        cv::minMaxLoc(smooth_texture, nullptr, &max_val);
        cv::Mat normalised_texture = smooth_texture / (max_val + 1e-6);

        // Canny edges
        cv::Mat grey_blur;
        // Using a Gaussian blur to reduce noise in the image
        cv::GaussianBlur(img_grey, grey_blur, cv::Size(3, 3), 0);

        cv::Mat canny_edges;
        cv::Canny(grey_blur, canny_edges, 30, 80);

        // Hough lines
        std::vector<cv::Vec4i> hough_lines;
        // Final three parameters are used to only select long and flat enough to be a valid horizon line
        cv::HoughLinesP(canny_edges, hough_lines, 1, CV_PI / 180, 30, 80, 10);

        int horizon = h / 2;
        cv::Mat hough_line_img = bgr_img.clone();
        bool horizon_found = false;

        // Iterating through the lines find and storing all of the lines found that are close to horizontal as they are more likely to be horizon lines
        if (!hough_lines.empty()) {
            std::vector<int> horizontal;
            for (auto& line : hough_lines) {
                int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
                double angle = std::abs(std::atan2((double)(y2 - y1), (double)(x2 - x1)) * 180.0 / CV_PI);
                if (angle < 10) {
                    horizontal.push_back((y1 + y2) / 2);
                    // cv::line(hough_line_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 1);
                }
            }

            // Finding the median of all potential horizon lines and using that as the horizon
            if (horizontal.size() > 3) {
                std::sort(horizontal.begin(), horizontal.end());
                size_t n = horizontal.size();
                if (n % 2 == 0) {
                    horizon = (horizontal[n / 2 - 1] + horizontal[n / 2]) / 2;
                } else {
                    horizon = horizontal[n / 2];
                }
                horizon_found = true;
            }
        }

        // Horizon position weights
        cv::Mat horizon_weights = cv::Mat::zeros(h, w, CV_32F);

        if (horizon_found) {
            // Ensuring that pixels are weighted stronger the higher above the horizon line they are as they are more likely to be part of the sky 
            for (int y = 0; y < horizon; y++) {
                float val = 1.0f - ((float)y / (float)horizon);
                horizon_weights.row(y).setTo(val);
            }
        } else {
            for (int y = 0; y < h; y++) {
                float val = std::max(0.0f, 1.0f - ((float)y / (float)(h * 0.6)));
                horizon_weights.row(y).setTo(val);
            }
        }

        // Colour gate and temporal weight
        cv::Mat colour_gate;
        // Normalising the colour mask into a set of probabilistic weights
        colour_mask.convertTo(colour_gate, CV_32F, 1.0 / 255.0);

        cv::Mat canny_f;
        // Gaining probabilistic weights from the canny edges
        canny_edges.convertTo(canny_f, CV_32F, 1.0 / 255.0);

        cv::Mat weighted_horizon, weighted_texture, colour_weight, edge_weight;

        if (horizon_found) {
            weighted_horizon = horizon_weights * 0.15;
            weighted_texture = (1.0 - normalised_texture) * 0.1;
            weighted_texture = weighted_texture.mul(colour_gate);
            colour_weight = colour_gate * 0.4;
            edge_weight = (1.0 - canny_f) * 0.35;
            edge_weight = edge_weight.mul(colour_gate);
        } else {
            weighted_horizon = horizon_weights * 0.05;
            weighted_texture = (1.0 - normalised_texture) * 0.15;
            weighted_texture = weighted_texture.mul(colour_gate);
            colour_weight = colour_gate * 0.4;
            edge_weight = (1.0 - canny_f) * 0.4;
            edge_weight = edge_weight.mul(colour_gate);
        }

        // Combine weights
        cv::Mat total_weight = colour_weight + weighted_texture + edge_weight + weighted_horizon;

        // Displaying the weighted image
        cv::Mat total_display;
        total_weight.convertTo(total_display, CV_8U, 255.0);
        cv::imshow("Weights", total_display);

        // Texture display
        double tex_max;
        cv::minMaxLoc(weighted_texture, nullptr, &tex_max);
        //if (tex_max > 0) {
        //    cv::Mat display_texture;
        //    weighted_texture.convertTo(display_texture, CV_8U, 255.0 / tex_max);
        //    cv::imshow("texture", display_texture);
        //}

        //cv::line(hough_line_img, cv::Point(0, horizon), cv::Point(w, horizon), cv::Scalar(0, 0, 255), 1);
        //cv::imshow("Horizon line", hough_line_img);
        //cv::imshow("Canny edges", canny_edges);
        cv::imshow("BGR img", bgr_img);
        //cv::imshow("colour mask", colour_mask);

        // Threshold
        cv::Mat threshold_img;
        cv::threshold(total_display, threshold_img, 140, 255, cv::THRESH_BINARY);

        // Close small holes within sky regions
        cv::Mat close_kernel = cv::Mat::ones(2, 2, CV_8U);
        cv::Mat closed_img;
        cv::morphologyEx(threshold_img, closed_img, cv::MORPH_CLOSE, close_kernel, cv::Point(-1, -1), 1);

        // Erosion / Dilation
        cv::Mat kernel = cv::Mat::ones(2, 2, CV_8U);
        cv::Mat eroded_img, dilated_img;
        cv::erode(closed_img, eroded_img, kernel, cv::Point(-1, -1), 1);
        cv::dilate(eroded_img, dilated_img, kernel, cv::Point(-1, -1), 2);
        //cv::imshow("Morphological operations", dilated_img);

        // Connected components
        cv::Mat labels, stats, centroids;
        // Finds each separate white blob in the dilated black and white image - each area of the ksy
        int num_labels = cv::connectedComponentsWithStats(dilated_img, labels, stats, centroids);

        cv::Mat clean_mask = cv::Mat::zeros(h, w, CV_8U);

        // Iterating through each area of the sky found
        for (int label = 1; label < num_labels; label++) {
            // Finding the area of this section of sky
            int area = stats.at<int>(label, cv::CC_STAT_AREA);
            // Finding the vertical position of the area of sky
            double centroid_y = centroids.at<double>(label, 1);

            // If the area of sky is so small then it is ignored as likely to be noise
            if (area < 150) {
                continue;
            }
            // Any area of sky in the lower quarter of the image then it is ignored as highly unlikely to be sky
            if (centroid_y > h * 0.6) {
                continue;
            }

            // Area of sky added to the image
            clean_mask.setTo(255, labels == label);
        }

        cv::imshow("Sky Detection", clean_mask);

        cv::Mat prob_colour;
        cv::applyColorMap(total_display, prob_colour, cv::COLORMAP_JET);
        cv::Mat overlay;
        cv::addWeighted(bgr_img, 0.6, prob_colour, 0.4, 0, overlay);
        cv::imshow("Sky Probability", overlay);

        if (cv::waitKey(100) == 'q') {
            break;
        }
    }

    capture.release();
    cv::destroyAllWindows();
    return 0;
}