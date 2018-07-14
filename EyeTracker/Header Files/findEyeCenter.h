using namespace System;
using namespace System::Runtime::InteropServices;

#include "opencv2/imgproc/imgproc.hpp"

#ifndef EYE_CENTER_H
#define EYE_CENTER_H

namespace EyeAPI
{
	public ref class EyeCenter
	{
	public:
		cv::Point findEyeCenter(cv::Mat face, cv::Rect eye, std::string debugWindow);
	};
}
#endif