#pragma once

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <queue>
#include <stdio.h>

using namespace cv;
using namespace std;

class centroOlho {
public:

	static Point encontrarCentro(Mat img, Rect result, int x, int Width);

};