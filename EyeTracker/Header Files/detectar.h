#pragma once

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <time.h>
#include <stdlib.h>

using namespace std;
using namespace cv;

class detectar {
public:

	static Rect Face(Mat& img, CascadeClassifier& cascade, CascadeClassifier& nestedCascade, double scale, bool tryflip);
	static vector<Rect> Olho(Mat img, CascadeClassifier& nestedCascade);
	static vector<Rect> OlhoProporcao(Rect face);

};
