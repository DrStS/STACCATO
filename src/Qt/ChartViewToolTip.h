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
 * \file ChartViewToolTip.h
 * This file holds the class of ChartViewToolTip.
 * \date 20/1/2018
 **************************************************************************************************/
#ifndef CHARTVIEWTOOLTIP_H
#define CHARTVIEWTOOLTIP_H

#include <QtCharts/QChartGlobal>
#include <QtWidgets/QGraphicsItem>
#include <QtGui/QFont>

class QGraphicsSceneMouseEvent;

QT_CHARTS_BEGIN_NAMESPACE
class QChart;
QT_CHARTS_END_NAMESPACE

QT_CHARTS_USE_NAMESPACE

class ChartViewToolTip : public QGraphicsItem {
public:
	ChartViewToolTip(QChart* parent);
	QRectF boundingRect() const;
	void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
	void setText(const QString &_text);
	void setAnchor(QPointF _point);
	void updateGeometry();

protected:
	void mousePressEvent(QGraphicsSceneMouseEvent *event);
	void mouseMoveEvent(QGraphicsSceneMouseEvent *event);

private:
	QChart* myChart;
	QPointF myAnchor;
	QRectF myRect;
	QString myText;
	QRectF myTextRect;
	QFont myFont;
};

#endif CHARTVIEWTOOLTIP_H