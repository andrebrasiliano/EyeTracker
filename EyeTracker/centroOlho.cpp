#include "Header Files/centroOlho.h"

int velocidade = 65;

Point unscalePoint(cv::Point p, cv::Rect origSize) {
	float ratio = (((float)velocidade) / origSize.width);
	int x = (int)round(p.x / ratio);
	int y = (int)round(p.y / ratio);
	return cv::Point(x, y);
}

Mat computeGradient(Mat mat) {

	Mat out(mat.rows, mat.cols, CV_64F);

	for (int y = 0; y < mat.rows; ++y) {
		const uchar *Mr = mat.ptr<uchar>(y);
		double *Or = out.ptr<double>(y);

		Or[0] = Mr[1] - Mr[0];
		for (int x = 1; x < mat.cols - 1; ++x) {
			Or[x] = (Mr[x + 1] - Mr[x - 1]) / 2.0;
		}
		Or[mat.cols - 1] = Mr[mat.cols - 1] - Mr[mat.cols - 2];
	}

	return out;
}

cv::Mat matrixMagnitude(Mat matX, Mat matY) {

	Mat mags(matX.rows, matX.cols, CV_64F);
	for (int y = 0; y < matX.rows; ++y) {
		const double *Xr = matX.ptr<double>(y), *Yr = matY.ptr<double>(y);
		double *Mr = mags.ptr<double>(y);
		for (int x = 0; x < matX.cols; ++x) {
			double gX = Xr[x], gY = Yr[x];
			double magnitude = sqrt((gX * gX) + (gY * gY));
			Mr[x] = magnitude;
		}
	}
	return mags;
}

double computeThreshold(Mat mat, double stdDevFactor) {
	cv::Scalar stdMagnGrad, meanMagnGrad;
	cv::meanStdDev(mat, meanMagnGrad, stdMagnGrad);
	double stdDev = stdMagnGrad[0] / sqrt(mat.rows*mat.cols);
	return stdDevFactor * stdDev + meanMagnGrad[0];
}

void possivelCentro(int x, int y, Mat weight, double gx, double gy, Mat out) {
	// for all possible centers
	for (int cy = 0; cy < out.rows; ++cy) {
		double *Or = out.ptr<double>(cy);
		const unsigned char *Wr = weight.ptr<unsigned char>(cy);
		for (int cx = 0; cx < out.cols; ++cx) {
			if (x == cx && y == cy) {
				continue;
			}
			// create a vector from the possible center to the gradient origin
			double dx = x - cx;
			double dy = y - cy;
			// normalize d
			double magnitude = sqrt((dx * dx) + (dy * dy));
			dx = dx / magnitude;
			dy = dy / magnitude;
			double dotProduct = dx*gx + dy*gy;
			dotProduct = std::max(0.0, dotProduct);
			// square and multiply by the weight

			Or[cx] += dotProduct * dotProduct * (Wr[cx] / 1.0F);

		}
	}
}

bool inMat(Point p, int rows, int cols) {
	return p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows;
}

bool floodShouldPushPoint(Point np, Mat mat) {
	return inMat(np, mat.rows, mat.cols);
}

Mat floodKillEdges(Mat mat) {
	rectangle(mat, cv::Rect(0, 0, mat.cols, mat.rows), 255);

	cv::Mat mask(mat.rows, mat.cols, CV_8U, 255);
	std::queue<cv::Point> toDo;
	toDo.push(cv::Point(0, 0));
	while (!toDo.empty()) {
		cv::Point p = toDo.front();
		toDo.pop();
		if (mat.at<float>(p) == 0.0f) {
			continue;
		}
		// add in every direction
		cv::Point np(p.x + 1, p.y); // right
		if (floodShouldPushPoint(np, mat)) toDo.push(np);
		np.x = p.x - 1; np.y = p.y; // left
		if (floodShouldPushPoint(np, mat)) toDo.push(np);
		np.x = p.x; np.y = p.y + 1; // down
		if (floodShouldPushPoint(np, mat)) toDo.push(np);
		np.x = p.x; np.y = p.y - 1; // up
		if (floodShouldPushPoint(np, mat)) toDo.push(np);
		// kill it
		mat.at<float>(p) = 0.0f;
		mask.at<uchar>(p) = 0;
	}
	return mask;
}


Point centroOlho::encontrarCentro(Mat img, Rect result, int x, int Width) {

	vector<Mat> rgbChannels(3);
	split(img, rgbChannels);
	Mat img_gray = rgbChannels[2];

	Rect rect = Rect(x, 0, Width, result.height);

	Mat regOlho = img_gray(result);
	regOlho = regOlho(rect);
	Mat roiOlho;

	resize(regOlho, roiOlho, cv::Size(velocidade, (int)(((float)velocidade) / regOlho.cols) * regOlho.rows));

	Mat gradientX = computeGradient(roiOlho);
	Mat gradientY = computeGradient(roiOlho.t()).t();

	Mat mags = matrixMagnitude(gradientX, gradientY);

	double gradientThresh = computeThreshold(mags, 50.0);

	for (int y = 0; y < roiOlho.rows; ++y) {
		double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
		const double *Mr = mags.ptr<double>(y);
		for (int x = 0; x < roiOlho.cols; ++x) {
			double gX = Xr[x], gY = Yr[x];
			double magnitude = Mr[x];
			if (magnitude > gradientThresh) {
				Xr[x] = gX / magnitude;
				Yr[x] = gY / magnitude;
			}
			else {
				Xr[x] = 0.0;
				Yr[x] = 0.0;
			}
		}
	}

	Mat weight;
	GaussianBlur(roiOlho, weight, cv::Size(5, 5), 0, 0);

	for (int y = 0; y < weight.rows; ++y) {
		unsigned char *row = weight.ptr<unsigned char>(y);
		for (int x = 0; x < weight.cols; ++x) {
			row[x] = (255 - row[x]);
		}
	}

	Mat outSum = Mat::zeros(roiOlho.rows, roiOlho.cols, CV_64F);

	for (int y = 0; y < weight.rows; ++y) {
		const double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
		for (int x = 0; x < weight.cols; ++x) {
			double gX = Xr[x], gY = Yr[x];
			if (gX == 0.0 && gY == 0.0) {
				continue;
			}
			possivelCentro(x, y, weight, gX, gY, outSum);
		}
	}

	double numGradients = (weight.rows*weight.cols);

	Mat out;
	outSum.convertTo(out, CV_32F, 1.0 / numGradients);

	//-- Find the maximum point
	Point maxP;
	double maxVal;
	minMaxLoc(out, NULL, &maxVal, NULL, &maxP);
	//-- Flood fill the edges

	Mat floodClone;
	//double floodThresh = computeDynamicThreshold(out, 1.5);
	double floodThresh = maxVal * 0.97F;
	threshold(out, floodClone, floodThresh, 0.0f, cv::THRESH_TOZERO);


	Mat mask = floodKillEdges(floodClone);

	// redo max
	minMaxLoc(out, NULL, &maxVal, NULL, &maxP, mask);

	Point ponto = unscalePoint(maxP, rect);
	ponto.x = result.x + rect.x + ponto.x;
	ponto.y = result.y + rect.y + ponto.y;

	return ponto;


}

