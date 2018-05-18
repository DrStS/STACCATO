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
#include "ChartView.h"
#include <QtGui/QMouseEvent>

ChartView::ChartView(QChart *chart, QWidget *parent) :
	QChartView(chart, parent),
	m_isTouching(false)
{
	setRubberBand(QChartView::RectangleRubberBand);
}

bool ChartView::viewportEvent(QEvent *event)
{
	if (event->type() == QEvent::TouchBegin) {
		// By default touch events are converted to mouse events. So
		// after this event we will get a mouse event also but we want
		// to handle touch events as gestures only. So we need this safeguard
		// to block mouse events that are actually generated from touch.
		m_isTouching = true;

		// Turn off animations when handling gestures they
		// will only slow us down.
		chart()->setAnimationOptions(QChart::NoAnimation);
	}
	return QChartView::viewportEvent(event);
}

void ChartView::mousePressEvent(QMouseEvent *_event)
{
	if (_event->buttons() == Qt::MiddleButton) {
		myPanStart = _event->localPos();
	}
	if (m_isTouching)
		return;
	QChartView::mousePressEvent(_event);
}

void ChartView::mouseMoveEvent(QMouseEvent *_event)
{
	if (_event->buttons() == Qt::MiddleButton) {
		this->setCursor(Qt::SizeAllCursor);
		QPointF numPixel = _event->localPos();
		chart()->scroll(myPanStart.x() - numPixel.x(), -(myPanStart.y() - numPixel.y()));
		myPanStart = numPixel;
	}
	if (m_isTouching)
		return;
	QChartView::mouseMoveEvent(_event);
}

void ChartView::mouseReleaseEvent(QMouseEvent *event)
{
	this->setCursor(Qt::ArrowCursor);
	if (m_isTouching)
		m_isTouching = false;

	QChartView::mouseReleaseEvent(event);
}

void ChartView::keyPressEvent(QKeyEvent *event)
{
	switch (event->key()) {
	case Qt::Key_Plus:
		chart()->zoomIn();
		break;
	case Qt::Key_Minus:
		chart()->zoomOut();
		break;
	case Qt::Key_Left:
		chart()->scroll(-10, 0);
		break;
	case Qt::Key_Right:
		chart()->scroll(10, 0);
		break;
	case Qt::Key_Up:
		chart()->scroll(0, 10);
		break;
	case Qt::Key_Down:
		chart()->scroll(0, -10);
		break;
	default:
		QGraphicsView::keyPressEvent(event);
		break;
	}
}

void ChartView::wheelEvent(QWheelEvent* _event) {
	QPoint numDegrees = _event->angleDelta() / 8;
	if(numDegrees.y()>0)
		chart()->zoomIn();
	else
		chart()->zoomOut();
}
