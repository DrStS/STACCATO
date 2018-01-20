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
#include "ChartViewToolTip.h"
#include <QtGui/QPainter>
#include <QtGui/QFontMetrics>
#include <QtWidgets/QGraphicsSceneMouseEvent>
#include <QtGui/QMouseEvent>
#include <QtCharts/QChart>

ChartViewToolTip::ChartViewToolTip(QChart *_chart) :QGraphicsItem(_chart), myChart(_chart){
}

QRectF ChartViewToolTip::boundingRect() const{
	QPointF anchor = mapFromParent(myChart->mapToPosition(myAnchor));
	QRectF rect;
	rect.setLeft(qMin(myRect.left(), anchor.x()));
	rect.setRight(qMax(myRect.right(), anchor.x()));
	rect.setTop(qMin(myRect.top(), anchor.y()));
	rect.setBottom(qMax(myRect.bottom(), anchor.y()));
	return rect;
}

void ChartViewToolTip::paint(QPainter *_painter, const QStyleOptionGraphicsItem *_option, QWidget *_widget){
	Q_UNUSED(_option)
		Q_UNUSED(_widget)
		QPainterPath path;
	path.addRoundedRect(myRect, 5, 5);

	QPointF anchor = mapFromParent(myChart->mapToPosition(myAnchor));
	if (!myRect.contains(anchor)) {
		QPointF point1, point2;

		// establish the position of the anchor point in relation to myRect
		bool above = anchor.y() <= myRect.top();
		bool aboveCenter = anchor.y() > myRect.top() && anchor.y() <= myRect.center().y();
		bool belowCenter = anchor.y() > myRect.center().y() && anchor.y() <= myRect.bottom();
		bool below = anchor.y() > myRect.bottom();

		bool onLeft = anchor.x() <= myRect.left();
		bool leftOfCenter = anchor.x() > myRect.left() && anchor.x() <= myRect.center().x();
		bool rightOfCenter = anchor.x() > myRect.center().x() && anchor.x() <= myRect.right();
		bool onRight = anchor.x() > myRect.right();

		// get the nearest myRect corner.
		qreal x = (onRight + rightOfCenter) * myRect.width();
		qreal y = (below + belowCenter) * myRect.height();
		bool cornerCase = (above && onLeft) || (above && onRight) || (below && onLeft) || (below && onRight);
		bool vertical = qAbs(anchor.x() - x) > qAbs(anchor.y() - y);

		qreal x1 = x + leftOfCenter * 10 - rightOfCenter * 20 + cornerCase * !vertical * (onLeft * 10 - onRight * 20);
		qreal y1 = y + aboveCenter * 10 - belowCenter * 20 + cornerCase * vertical * (above * 10 - below * 20);;
		point1.setX(x1);
		point1.setY(y1);

		qreal x2 = x + leftOfCenter * 20 - rightOfCenter * 10 + cornerCase * !vertical * (onLeft * 20 - onRight * 10);;
		qreal y2 = y + aboveCenter * 20 - belowCenter * 10 + cornerCase * vertical * (above * 20 - below * 10);;
		point2.setX(x2);
		point2.setY(y2);

		path.moveTo(point1);
		path.lineTo(anchor);
		path.lineTo(point2);
		path = path.simplified();
	}
	_painter->setBrush(QColor(255, 255, 255));
	_painter->drawPath(path);
	_painter->drawText(myTextRect, myText);
}

void ChartViewToolTip::mousePressEvent(QGraphicsSceneMouseEvent *event){
	event->setAccepted(true);
}

void ChartViewToolTip::mouseMoveEvent(QGraphicsSceneMouseEvent *event){
	if (event->buttons() & Qt::LeftButton) {
		setPos(mapToParent(event->pos() - event->buttonDownPos(Qt::LeftButton)));
		event->setAccepted(true);
	}
	else {
		event->setAccepted(false);
	}
}

void ChartViewToolTip::setText(const QString &_text){
	myText = _text;
	QFontMetrics metrics(myFont);
	myTextRect = metrics.boundingRect(QRect(0, 0, 150, 150), Qt::AlignLeft, myText);
	myTextRect.translate(5, 5);
	prepareGeometryChange();
	myRect = myTextRect.adjusted(-5, -5, 5, 5);
}

void ChartViewToolTip::setAnchor(QPointF _point){
	myAnchor = _point;
}

void ChartViewToolTip::updateGeometry(){
	prepareGeometryChange();
	setPos(myChart->mapToPosition(myAnchor) + QPoint(10, -50));
}