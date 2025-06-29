//
// Created by Chlebus, Grzegorz on 25.08.17.
// Copyright (c) Chlebus, Grzegorz. All rights reserved.
//

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

class Line
{
public:
  Line();
  
  Line(cv::Point2f point, cv::Point2f vector);

  static Line fromRhoTheta(float rho, float theta);

  static Line fromTwoPoints(cv::Point2f p1, cv::Point2f p2);

  bool computeIntersectionPoint(const Line& line, cv::Point2f& intersectionPoint) const;

  cv::Point2f getPoint() const;

  cv::Point2f getVector() const;

  float getDistance(const cv::Point2f& point) const;

  float getPerpendicularDistance(const cv::Point2f& point) const;

  float evaluateByX(float x) const;
  
  bool isDuplicate(const Line& otherLine) const;

  bool isParallel(const Line& otherLine, double tol = BIG_EPS) const;

  void toImplicit(cv::Point2f& n, float& c) const;

  double getAngle(const Line& otherLine) const;

  cv::Point2f getPointOnLineClosestTo(const cv::Point2f point) const;

  cv::Point2f u; // point on line
  cv::Point2f v; // collinear vector

  constexpr static double EPS = 1e-5;
  constexpr static double BIG_EPS = 1e-2;
  constexpr static double PIXEL_EPS = 10;
};
