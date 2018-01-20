/*  Copyright &copy; 2018, Dr. Stefan Sicklinger, Munich \n
 *
 *  All rights reserved.
 *
 *  This file is part of STACCATO.
 *
 *  STACCATO is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  STACCATO is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with STACCATO.  If not, see http://www.gnu.org/licenses/.
 */
/***********************************************************************************************//**
 * \file ChartView.h
 * This file holds the class of ChartView.
 * \date 20/1/2018
 **************************************************************************************************/
#ifndef CHARTVIEW_H
#define CHARTVIEW_H

#include <QtCharts/QChartView>
#include <QtWidgets/QRubberBand>

QT_CHARTS_USE_NAMESPACE

class ChartView : public QChartView
{
public:
	ChartView(QChart *chart, QWidget *parent = 0);

protected:
	bool viewportEvent(QEvent *event);
	void mousePressEvent(QMouseEvent *_event);
	void mouseMoveEvent(QMouseEvent *_event);
	void mouseReleaseEvent(QMouseEvent *event);
	void keyPressEvent(QKeyEvent *event);
	void wheelEvent(QWheelEvent* _event);

private:
	bool m_isTouching;
	QPointF myPanStart;
};

#endif