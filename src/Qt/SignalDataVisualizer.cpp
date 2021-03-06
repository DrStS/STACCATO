/*  Copyright &copy; 2016, Stefan Sicklinger, Munich
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
#include "SignalDataVisualizer.h"
#include "FieldDataVisualizer.h"
#include "ChartViewToolTip.h"
#include "HMesh.h"
#include "ChartView.h"
#include "OutputDatabase.h"
#include "VisualizerSetting.h"
#include "VectorFieldResults.h"
//QT5
#include <QTreeView>
#include <QDockWidget>
#include <QtWidgets>
#include <QLabel>
#include <QPushButton>
#include <QLineEdit>
#include <QComboBox>
#include <QCheckBox>
#include <QGroupBox>
#include <QFormLayout>
#include <QSpinBox>
#include <QtCharts/QLineSeries>
#include <QtCharts/QChartView>
#include <QSplineSeries>
#include <QRadioButton>
#include <QList>
#include <QPointF>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QValueAxis>
#include <QScatterSeries>
#include <QLegendMarker>
#include <QLogValueAxis>
#include <QGesture>

QT_CHARTS_USE_NAMESPACE

SignalDataVisualizer::SignalDataVisualizer()
{
	
}

SignalDataVisualizer::~SignalDataVisualizer() {
}

void SignalDataVisualizer::initiate() {
	setWindowIcon(QIcon(":/Qt/resources/STACCATO.png"));
	setWindowTitle("STACCATO Visualizer");
	createActions();
	createMenus();
	createDockWindow();
	// Chart Properties
	myChart2D = new QChart();
	myAxisX = new QValueAxis();
	myAxisY = new QValueAxis();

	chartView2D = new ChartView(myChart2D);
	chartView2D->setRubberBand(QChartView::RectangleRubberBand);	// For Rectangle Draw Zoom

	mySeries2scatter = new QScatterSeries();						// Selection Points
	mySnapSeries2scatter = new QScatterSeries();					// Snapped Point

	myAxisLogX = new QLogValueAxis();								// Logarthmic X-Axis

	createChart();

	setCentralWidget(chartView2D);
}

void SignalDataVisualizer::attachSubject(FieldDataVisualizer* _fieldDataVisualizer) {
	myFieldDataVisualizerSubject = _fieldDataVisualizer;
	_fieldDataVisualizer->attachObserver(this);
}

void SignalDataVisualizer::update() {
	if (myFieldDataVisualizerSubject->myCurrentPickerType == STACCATO_Picker_Node)
		setSelection(myFieldDataVisualizerSubject->mySelectedNodes);
	else if (myFieldDataVisualizerSubject->myCurrentPickerType == STACCATO_Picker_Element)
		setSelection(myFieldDataVisualizerSubject->mySelectedElements);
}

void SignalDataVisualizer::createMenus(void)
{
	myFileMenu = menuBar()->addMenu(tr("&File"));
	myFileMenu->addAction(myExitAction);

	myHelpMenu = menuBar()->addMenu(tr("&Help"));
	myHelpMenu->addAction(myAboutAction);
}

void SignalDataVisualizer::createActions(void)
{
	// File actions
	myExitAction = new QAction(tr("Exit"), this);
	myExitAction->setShortcut(tr("Ctrl+Q"));
	myExitAction->setIcon(QIcon(":/Qt/resources/closeDoc.png"));
	myExitAction->setStatusTip(tr("Exit the application"));
	connect(myExitAction, SIGNAL(triggered()), this, SLOT(close()));
	
	//Help actions
	myAboutAction = new QAction(tr("About"), this);
	myAboutAction->setStatusTip(tr("About the application"));
	myAboutAction->setIcon(QIcon(":/Qt/resources/about.png"));
	connect(myAboutAction, SIGNAL(triggered()), this, SLOT(about()));
}

void SignalDataVisualizer::about() {
	QMessageBox::about(this, tr("About STACCATO - Visualizer"),
		tr("<h2>STACCATO: STefAn's Computational vibroaCoustics Analysis TOol</h2>"
			"<p>Copyright &copy; 2017 "
			"<p>STACCATO Visualizer is using Qt and QCharts."));
}

void SignalDataVisualizer::createChart() {
	// Add a Sample Plot
	if (getSelection().size() != 0) {
		for (int i = 0; i < getSelection().size(); i++) {
			addChildToTree(myHistoryRoot, QString::fromStdString("Ux_Re at Node " + std::to_string(getSelection()[i])), false);
			add2dPlotToChart(getSelection()[i], STACCATO_x_Re);
		}
		myOutputTree->expandAll();
	}
}

void SignalDataVisualizer::add2dPlotToChart(int _nodeLabel, STACCATO_VectorField_components _type) {
	// Preparing Plot Data
	xValues.clear();
	yValues.clear();

	for (int it = 0; it < myHMesh->myOutputDatabase->getTimeDescription(myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX).size(); it++) {
		if (myHMesh->myOutputDatabase->myAnalyses[myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX].type == STACCATO_Analysis_Static)
			xValues.push_back(0);
		else
			xValues.push_back(std::stoi(myHMesh->myOutputDatabase->getTimeDescription(myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX)[it]));
	}

	int nodeIndex = myHMesh->convertNodeLabelToNodeIndex(_nodeLabel);
	for (int i = 0; i < xValues.size(); i++) {
		yValues.push_back(myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX].getResultScalarFieldAtNodes(_type, i)[nodeIndex]);
	}
	// Data Series
	mySeries2D = new QLineSeries();
	for (int i = 0; i < xValues.size(); i++) {
		mySeries2D->append(xValues[i], yValues[i]);;
	}
	if (_type == STACCATO_x_Re) {
		mySeries2D->setName(tr("Ux_Re Node ") + QString::fromStdString(std::to_string(_nodeLabel)));
	}
	else if (_type == STACCATO_y_Re) {
		mySeries2D->setName(tr("Uy_Re Node ") + QString::fromStdString(std::to_string(_nodeLabel)));
	}
	else if (_type == STACCATO_z_Re) {
		mySeries2D->setName(tr("Uz_Re Node ") + QString::fromStdString(std::to_string(_nodeLabel)));
	}
	else if (_type == STACCATO_x_Im) {
		mySeries2D->setName(tr("Ux_Im Node ") + QString::fromStdString(std::to_string(_nodeLabel)));
	}
	else if (_type == STACCATO_y_Im) {
		mySeries2D->setName(tr("Uy_Im Node ") + QString::fromStdString(std::to_string(_nodeLabel)));
	}
	else if (_type == STACCATO_z_Im) {
		mySeries2D->setName(tr("Uz_Im Node ") + QString::fromStdString(std::to_string(_nodeLabel)));
	}
	mySeries2D->setPointsVisible(myPointVisibilityBox->isChecked());

	mySeriesList.append(mySeries2D);

	myChart2D->addSeries(mySeriesList.last());
	myChart2D->setTitle("Solution Distribution over Node");
	myChart2D->legend()->setAlignment(Qt::AlignBottom);				// Legend Alignment

	QPen pen(mySeriesList.last()->color());							// Line Width
	pen.setWidth(myLineWidthText->text().toDouble());
	mySeriesList.last()->setPen(pen);

	myAxisX->setTitleText("Frequency (in Hz)");
	myAxisX->setLabelFormat("%d");
	//myAxisX->setMinorTickCount(0);
	myAxisX->setTickCount(10);

	myAxisLogX->setTitleText("Frequency (in Hz), Log Scale");
	myAxisLogX->setLabelFormat("%g");
	myAxisLogX->setBase(10.0);
	myAxisLogX->setMinorTickCount(-1);

	// Chart Axes
	if (myHMesh->myOutputDatabase->myAnalyses[myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX].type == STACCATO_Analysis_Static)
	{
		myAxisX->setTitleText("Static");
		myAxisLogX->setTitleText("Static, Log Scale");
	}

	if (_type == STACCATO_x_Re || _type == STACCATO_y_Re || _type == STACCATO_z_Re || _type == STACCATO_x_Im || _type == STACCATO_y_Im || _type == STACCATO_z_Im) {
		myAxisY->setTitleText("Displacement");
	}
	else {
		myAxisY->setTitleText("Unrecognized Axis");
	}
	myAxisY->setLabelFormat("%.2E");

	// Set Connections
	connect(mySeriesList.last(), &QLineSeries::clicked, this, &SignalDataVisualizer::handleClickedPoint);
	connect(mySeriesList.last(), &QLineSeries::hovered, this, &SignalDataVisualizer::tooltip);

	enableInteractiveClick(mySeriesList.last());		// Enable Scattering

	// Axes-Chart Association
	myChart2D->addAxis(myAxisX, Qt::AlignBottom);
	myChart2D->setAcceptHoverEvents(true);
	myChart2D->addAxis(myAxisY, Qt::AlignLeft);

	chartView2D->setChart(myChart2D);
	chartView2D->setRenderHint(QPainter::Antialiasing);
	chartView2D->setMinimumSize(600, 300);

	myAxisX->setTickCount(mySeriesList.last()->count());
	mySeriesList.last()->attachAxis(myAxisX);
	//myAxisY->setTickCount(mySeries2D->count());
	mySeriesList.last()->attachAxis(myAxisY);
	mySeries2scatter->attachAxis(myAxisX);
	mySeries2scatter->attachAxis(myAxisY);
	mySnapSeries2scatter->attachAxis(myAxisX);
	mySnapSeries2scatter->attachAxis(myAxisY);

	updateLegends();

	// Coordinate Hover
	myCoordX = new QGraphicsSimpleTextItem(myChart2D);
	myCoordX->setPos(myChart2D->size().width() / 2 - 50, myChart2D->size().height());
	myCoordY = new QGraphicsSimpleTextItem(myChart2D);
	myCoordY->setPos(myChart2D->size().width() / 2 + 50, myChart2D->size().height());

	double yValueMinNew = *std::min_element(yValues.begin(), yValues.end());
	double yValueMaxNew = *std::max_element(yValues.begin(), yValues.end());
	
	autoScale(yValueMinNew, yValueMaxNew);
	if(myLogXAxisBox->isChecked()){
		myLogScaleTriggered();
	}
}

void SignalDataVisualizer::updateLegends(){
	const auto markers = myChart2D->legend()->markers();
	for (QLegendMarker *marker : markers) {
		if (marker->series()->name().toStdString() == "DeleteSeriesLegend") {
			marker->setVisible(false);
		}
	}
}
	
void SignalDataVisualizer::autoScale(double _yValueMinNew, double _yValueMaxNew) {
	// Update Properties
	double margin = 0.1;
	myMinX->setText(QString::fromStdString(std::to_string(*std::min_element(xValues.begin(), xValues.end()))));
	myMaxX->setText(QString::fromStdString(std::to_string(*std::max_element(xValues.begin(), xValues.end()))));
	
	myRangeWidgetYUpdate(*std::min_element(yValues.begin(), yValues.end()) - margin, *std::max_element(yValues.begin(), yValues.end()) + margin);

	if(myMinX->text().toDouble()!= myMaxX->text().toDouble())
		myAxisX->setRange(myMinX->text().toDouble(), myMaxX->text().toDouble());
	else
		myAxisX->setRange(myMinX->text().toDouble()-10, myMaxX->text().toDouble()+10);

	myAxisY->setRange(myMinY->text().toDouble(), myMaxY->text().toDouble());
	myAxisX->setTickCount(10);
	static bool autoScaleActive = false;
	if (autoScaleActive) {
		if (myValueMinY < _yValueMinNew && myValueMaxY < _yValueMaxNew) {
			myAxisY->setRange(myValueMinY - margin, _yValueMaxNew + margin);
			myValueMaxY = _yValueMaxNew;
		}
		else if (myValueMinY > _yValueMinNew && myValueMaxY < _yValueMaxNew) {
			myAxisY->setRange(_yValueMinNew - margin, _yValueMaxNew + margin);
			myValueMinY = _yValueMinNew;
			myValueMaxY = _yValueMaxNew;
		}
		else if (myValueMinY < _yValueMinNew && myValueMaxY > _yValueMaxNew) {
			myAxisY->setRange(myValueMinY - margin, myValueMaxY + margin);
		}
		else if (myValueMinY > _yValueMinNew && myValueMaxY > _yValueMaxNew) {
			myAxisY->setRange(_yValueMinNew - margin, myValueMaxY + margin);
			myValueMinY = _yValueMinNew;
		}
		myRangeWidgetYUpdate(myValueMinY - margin, myValueMaxY + margin);
	}
	else {
		myValueMinY = *std::min_element(yValues.begin(), yValues.end());
		myValueMaxY = *std::max_element(yValues.begin(), yValues.end());
	}
	autoScaleActive = true;

	updateAxesConnections();
}

void SignalDataVisualizer::updateAxesConnections(void) {
	connect(myChart2D->axisX(), SIGNAL(rangeChanged(qreal, qreal)), this, SLOT(myRangeWidgetXUpdate(qreal, qreal)));
	connect(myChart2D->axisY(), SIGNAL(rangeChanged(qreal, qreal)), this, SLOT(myRangeWidgetYUpdate(qreal, qreal)));
}

void SignalDataVisualizer::myRangeWidgetXUpdate(qreal _min, qreal _max) {
	myMinX->setText(QString::number(_min));
	myMaxX->setText(QString::number(_max));
}

void SignalDataVisualizer::myRangeWidgetYUpdate(qreal _min, qreal _max) {
	myMinY->setText(QString::number(_min));
	myMaxY->setText(QString::number(_max));
}

void SignalDataVisualizer::enableInteractiveClick(QLineSeries* _lineSeries) {
	mySeries2scatter->setName(tr("DeleteSeriesLegend"));
	mySeries2scatter->setMarkerSize(15.0);
	myChart2D->addSeries(mySeries2scatter);
	mySnapSeries2scatter->setName(tr("DeleteSeriesLegend"));
	mySnapSeries2scatter->setMarkerSize(15.0);
	connect(mySnapSeries2scatter, &QScatterSeries::hovered, this, &SignalDataVisualizer::tooltip);
	myChart2D->addSeries(mySnapSeries2scatter);
}

void SignalDataVisualizer::handleClickedPoint(const QPointF &_point) {
	QPointF clickedPoint = _point;
	// Find the closest point from series 1
	QPointF closest(INT_MAX, INT_MAX);
	qreal distance(INT_MAX);
	
	foreach (QLineSeries* series, mySeriesList) {
		const auto points = series->points();
		for (const QPointF &currentPoint : points) {
			qreal currentDistance = qSqrt((currentPoint.x() - clickedPoint.x())
				* (currentPoint.x() - clickedPoint.x())
				+ (currentPoint.y() - clickedPoint.y())
				* (currentPoint.y() - clickedPoint.y()));
			if (currentDistance < distance) {
				distance = currentDistance;
				closest = currentPoint;
			}
		}
	}

	// Append the Closest or Interpolated (Current) Point
	if (myInterpolatingBox->isChecked()) {
		if (hoverFlag == 0) {
			mySeries2scatter->append(clickedPoint);
			myXCoordInfo->setText(QString::number(clickedPoint.x()));
			myYCoordInfo->setText(QString::number(clickedPoint.y()));
		}
		else {
			mySnapSeries2scatter->clear();
			mySnapSeries2scatter->append(clickedPoint);
		}
	}
	else {
		if (hoverFlag == 0) {
			mySeries2scatter->append(closest);
			myXCoordInfo->setText(QString::number(closest.x()));
			myYCoordInfo->setText(QString::number(closest.y()));
		}
		else {
			mySnapSeries2scatter->clear();
			mySnapSeries2scatter->append(closest);
		}
	}

	// Tooltip Routine
	if (myToolTipBox->isChecked()) {
		static bool myToolTipActive = false; 

		ChartViewToolTip *myToolTipT = new ChartViewToolTip(myChart2D);
		if (myInterpolatingBox->isChecked()) {
			myToolTipT->setText(QString("X: %1 \nY: %2 ").arg(clickedPoint.x()).arg(clickedPoint.y()));
			myToolTipT->setAnchor(clickedPoint);
		}
		else {
			myToolTipT->setText(QString("X: %1 \nY: %2 ").arg(closest.x()).arg(closest.y()));
			myToolTipT->setAnchor(closest);
		}

		myToolTipT->setZValue(11);
		myToolTipT->updateGeometry();
		if (!mySnapOnHoverBox->isChecked()) {
			myCallOuts.append(myToolTipT);
			myCallOuts.last()->show();
			myCallOuts.last()->updateGeometry();
		}
		else {
			foreach (ChartViewToolTip *callout, mySnapCallOuts) {
				callout->scene()->removeItem(callout);
				mySnapCallOuts.pop_back();
			}
			mySnapCallOuts.append(myToolTipT);
			mySnapCallOuts.last()->show();
			mySnapCallOuts.last()->updateGeometry();
		}
	}
	chartView2D->repaint();
}

void SignalDataVisualizer::tooltip(QPointF _point, bool _state) {
	static bool myToolTipActive = false;
	if (!myToolTipActive) {
		myToolTip = new ChartViewToolTip(myChart2D);
		myToolTipActive = true;
	}

	if (_state) {
		myToolTip->setText(QString("X: %1 \nY: %2 ").arg(_point.x()).arg(_point.y()));
		myToolTip->setAnchor(_point);
		myToolTip->setZValue(11);
		myToolTip->updateGeometry();
		myToolTip->show();
		if (mySnapOnHoverBox->isChecked()) {
			myToolTip->hide();
			hoverFlag = 1;					// Snap Flag Set
			handleClickedPoint(_point);
			hoverFlag = 0;					// Snap Flag Reset
		}
	}
	else {
		myToolTip->hide();
		if(mySnapOnHoverBox->isChecked()){
			if (myToolTipBox->isChecked()) {
				foreach (ChartViewToolTip *callout , mySnapCallOuts) {
					callout->scene()->removeItem(callout);
					mySnapCallOuts.pop_back();
				}
			}
		}
		chartView2D->repaint();
	}

}

QTreeWidgetItem* SignalDataVisualizer::addRootToTree(QTreeWidget* _tree, QString _name, bool _checkable) {
	QTreeWidgetItem* item = new QTreeWidgetItem(_tree);
	item->setText(0, _name);
	item->setFlags(item->flags() & ~Qt::ItemIsSelectable);
	if (_checkable) {
		item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
		item->setCheckState(0, Qt::Unchecked);
	}
	_tree->addTopLevelItem(item);
	return item;
}

void SignalDataVisualizer::addChildToTree(QTreeWidgetItem* _parent, QString _name, bool _checkable) {
	QTreeWidgetItem* item = new QTreeWidgetItem();
	item->setText(0, _name);
	if (_checkable) {
		item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
		item->setCheckState(0, Qt::Unchecked);
	}
	_parent->addChild(item);
}

void SignalDataVisualizer::prepareTreeMenu(const QPoint& _pos){
	QMenu* treeMenu = new QMenu();

	myAddAxisAction = new QAction(tr("Add as new axis"), this);
	connect(myAddAxisAction, SIGNAL(triggered()), this, SLOT(addOrdinateToSeries()));

	myDeleteAction = new QAction(tr("Delete series"), this);
	connect(myDeleteAction, SIGNAL(triggered()), this, SLOT(deleteSeries()));

	treeMenu->addAction(myAddAxisAction);
	treeMenu->addAction(myDeleteAction);

	QPoint pointAt(_pos);
	treeMenu->exec(myOutputTree->mapToGlobal(_pos));
}

void SignalDataVisualizer::createDockWindow(void) {
	myOutputTreeDock = new QDockWidget(tr("Output Tree"), this);
	myOutputTreeDock->setAllowedAreas(Qt::LeftDockWidgetArea);

	myOutputTree = new QTreeWidget;
	myOutputTree->setColumnCount(1);
	myOutputTree->header()->close();
	myOutputTree->setContextMenuPolicy(Qt::CustomContextMenu);
	connect(myOutputTree, &QTreeWidget::customContextMenuRequested, this, &SignalDataVisualizer::prepareTreeMenu);

	myPickTree = new QTreeWidget;
	myPickTree->setColumnCount(1);
	myPickTree->header()->close();

	myHistoryRoot = addRootToTree(myOutputTree, tr("History Outputs"), false);

	myNodePickLabel = new QLabel(this);
	myNodePickLabel->setText(tr("Enter ID:"));

	myNodePickText = new QLineEdit(this);
	connect(myNodePickText, SIGNAL(returnPressed()), this, SLOT(updateList()));

	myPickListButton = new QPushButton(this);
	myPickListButton->setIcon(QIcon(":/Qt/resources/list.ico"));
	myPickListButton->setStatusTip(tr("List the picked Node/ Element Variables"));
	myPickListButton->setFlat(true);
	connect(myPickListButton, SIGNAL(clicked()), this, SLOT(updateList()));

	myPickerButton = new QPushButton(this);
	myPickerButton->setIcon(QIcon(":/Qt/resources/picker.ico"));
	myPickerButton->setStatusTip(tr("Enable Picker"));
	myPickerButton->setCheckable(true);
	myPickerButton->setFlat(true);

	myPickAddButton = new QPushButton(this);
	myPickAddButton->setIcon(QIcon(":/Qt/resources/add.ico"));
	myPickAddButton->setStatusTip(tr("Add the picked Node/ Element Variables"));
	myPickAddButton->setFlat(true);
	connect(myPickAddButton, SIGNAL(clicked()), this, SLOT(updateOutputTree()));

	myNodePickRadio = new QRadioButton("Node", this);
	myNodePickRadio->setChecked(true);
	myElementPickRadio = new QRadioButton("Element", this);
	myForceNodeRadio = new QRadioButton("Element Nodal", this);
	myForceIntPointsRadio = new QRadioButton("Element IPs", this);

	mySaveButton = new QPushButton("Save", this);
	connect(mySaveButton, SIGNAL(clicked()), this, SLOT(saveChart()));

	myResetButton = new QPushButton("Reset", this);
	connect(myResetButton, SIGNAL(clicked()), this, SLOT(resetChart()));
	QGridLayout *layoutTemp = new QGridLayout;
	layoutTemp->addWidget(myOutputTree, 1, 1, 1, -1, Qt::AlignLeft);
	layoutTemp->addWidget(myNodePickLabel, 2, 1);
	layoutTemp->addWidget(myNodePickText, 2, 2, 1, 2);
	layoutTemp->addWidget(myPickListButton, 2, 4);
	layoutTemp->addWidget(myNodePickRadio, 3, 2);
	layoutTemp->addWidget(myElementPickRadio, 3, 3);
	layoutTemp->addWidget(myPickerButton, 3, 4);
	layoutTemp->addWidget(myForceNodeRadio, 4, 2);
	layoutTemp->addWidget(myForceIntPointsRadio, 4, 3);
	layoutTemp->addWidget(myPickAddButton, 4, 4);
	layoutTemp->addWidget(myPickTree, 5, 1, 1, -1, Qt::AlignLeft);
	layoutTemp->addWidget(mySaveButton, 6, 1, 1, 2, Qt::AlignCenter);
	layoutTemp->addWidget(myResetButton, 6, 3, 1, 2, Qt::AlignCenter);
	QWidget* temp = new QWidget(this);
	temp->setLayout(layoutTemp);
	
	myOutputTreeDock->setWidget(temp);
	myOutputTreeDock->show();

	addDockWidget(Qt::LeftDockWidgetArea, myOutputTreeDock);
	// Chart Properties
	myChartPropertiesDock = new QDockWidget(tr("Chart Properties"), this);
	myChartPropertiesDock->setAllowedAreas(Qt::RightDockWidgetArea);

	myRangeLabel = new QLabel(this);
	myRangeLabel->setText("<b>Range</b>");

	myMinXLabel = new QLabel(this);
	myMinXLabel->setText(tr("Min X:"));
	myMinX = new QLineEdit(this);
	myMinX->setFixedWidth(70);
	connect(myMinX, SIGNAL(editingFinished()), this, SLOT(myChartRangeUpdate()));

	myMaxXLabel = new QLabel(this);
	myMaxXLabel->setText(tr("Max X:"));
	myMaxX = new QLineEdit(this);
	myMaxX->setFixedWidth(70);
	connect(myMaxX, SIGNAL(editingFinished()), this, SLOT(myChartRangeUpdate()));

	myMinYLabel = new QLabel(this);
	myMinYLabel->setText(tr("Min Y:"));
	myMinY = new QLineEdit(this);
	myMinY->setFixedWidth(70);
	connect(myMinY, SIGNAL(editingFinished()), this, SLOT(myChartRangeUpdate()));

	myMaxYLabel = new QLabel(this);
	myMaxYLabel->setText(tr("Max Y:"));
	myMaxY = new QLineEdit(this);
	myMaxY->setFixedWidth(70);
	connect(myMaxY, SIGNAL(editingFinished()), this, SLOT(myChartRangeUpdate()));

	myLogXAxisBox = new QCheckBox("x_Log", this);
	connect(myLogXAxisBox, SIGNAL(clicked()), this, SLOT(myLogScaleTriggered()));

	myLogYAxisBox = new QCheckBox("y_Log", this);
	myLogYAxisBox->setEnabled(false);
	connect(myLogXAxisBox, SIGNAL(clicked()), this, SLOT(myLogScaleTriggered()));

	myClearSelectionButton = new QPushButton(this);
	myClearSelectionButton->setText(tr("Clear Selection"));
	myClearSelectionButton->setStatusTip(tr("Clear selected Points"));
	connect(myClearSelectionButton, SIGNAL(clicked()), this, SLOT(myClearPointSelection()));

	mySelectionLabel = new QLabel(this);
	mySelectionLabel->setText("<b>Selection</b>");

	myXCoordInfo = new QLineEdit(this);
	myXCoordInfo->setReadOnly(true);
	myXCoordInfo->setFixedWidth(70);
	myXCoordInfo->setAlignment(Qt::AlignLeft);

	myYCoordInfo = new QLineEdit(this);
	myYCoordInfo->setReadOnly(true);
	myYCoordInfo->setFixedWidth(70);
	myYCoordInfo->setAlignment(Qt::AlignLeft);

	myXCoordInfoLabel = new QLabel(this);
	myXCoordInfoLabel->setText("x:");
	myXCoordInfoLabel->setAlignment(Qt::AlignRight);

	myYCoordInfoLabel = new QLabel(this);
	myYCoordInfoLabel->setText("y:");
	myYCoordInfoLabel->setAlignment(Qt::AlignRight);

	myInterpLabel = new QLabel(this);
	myInterpLabel->setText("<b>Hovering</b>");

	myInterpolatingBox = new QCheckBox(this);
	myInterpolatingBox->setText("Interpolation");

	myToolTipBox = new QCheckBox(this);
	myToolTipBox->setText("Tool Tip");
	myToolTipBox->setChecked(true);
	connect(myToolTipBox, SIGNAL(clicked()), this, SLOT(updateToolTip()));

	mySnapOnHoverBox = new QCheckBox("Snap to Series", this);
	mySnapOnHoverBox->setChecked(true);
	connect(mySnapOnHoverBox, SIGNAL(clicked()), this, SLOT(updateSnap()));

	myChartPropertyLabel = new QLabel(this);
	myChartPropertyLabel->setText("<b>Property</b>");

	myPointVisibilityBox = new QCheckBox(this);
	myPointVisibilityBox->setText("Data Points");
	myPointVisibilityBox->setChecked(true);
	connect(myPointVisibilityBox, SIGNAL(clicked()), this, SLOT(updateSeriesProperty()));

	myLineWidthBox = new QCheckBox(this);
	myLineWidthBox->setText("Line Width");
	connect(myLineWidthBox, SIGNAL(clicked()), this, SLOT(setLineWidthWidgets()));

	myLineWidthLabel = new QLabel(this);
	myLineWidthLabel->setText("Set:");
	myLineWidthLabel->setEnabled(false);

	myLineWidthText = new QLineEdit(this);
	myLineWidthText->setFixedWidth(70);
	myLineWidthText->setAlignment(Qt::AlignLeft);
	myLineWidthText->setText(tr("2.0"));
	myLineWidthText->setEnabled(false);
	connect(myLineWidthText, SIGNAL(textChanged(const QString &)), this, SLOT(updateSeriesProperty()));

	mySetColorBox = new QCheckBox("Color", this);
	connect(mySetColorBox, SIGNAL(clicked()), this, SLOT(myColorSelectionTriggered()));

	myColorText = new QLineEdit(this);
	myColorText->setFixedWidth(70);
	myColorText->setEnabled(false);
	myColorText->setReadOnly(true);

	myPickColorButton = new QPushButton("Pick", this);
	myPickColorButton->setFixedWidth(30);
	myPickColorButton->setEnabled(false);
	connect(myPickColorButton, SIGNAL(clicked()), this, SLOT(myColorPickWidget()));

	myViewLabel = new QLabel("<b>Options</b>", this);

	myResetZoomButton = new QPushButton("Reset Zoom", this);
	connect(myResetZoomButton, SIGNAL(clicked()), this, SLOT(myResetChart()));

	QGridLayout *layoutDock = new QGridLayout;
	layoutDock->addWidget(myRangeLabel, 0, 1);
	layoutDock->addWidget(myMinXLabel, 1, 1);
	layoutDock->addWidget(myMinX, 1, 2);
	layoutDock->addWidget(myMaxXLabel, 2, 1);
	layoutDock->addWidget(myMaxX, 2, 2);
	layoutDock->addWidget(myMinYLabel, 1, 3);
	layoutDock->addWidget(myMinY, 1, 4);
	layoutDock->addWidget(myMaxYLabel, 2, 3);
	layoutDock->addWidget(myMaxY, 2, 4);
	layoutDock->addWidget(myLogXAxisBox, 3, 2);
	layoutDock->addWidget(myLogYAxisBox, 3, 4);

	layoutDock->addWidget(mySelectionLabel, 4, 1);
	layoutDock->addWidget(myXCoordInfoLabel, 5, 1);
	layoutDock->addWidget(myXCoordInfo, 5, 2);
	layoutDock->addWidget(myYCoordInfoLabel, 6, 1);
	layoutDock->addWidget(myYCoordInfo, 6, 2);
	layoutDock->addWidget(myClearSelectionButton, 6, 3, 1, 2, Qt::AlignLeft);

	layoutDock->addWidget(myInterpLabel, 7, 1);
	layoutDock->addWidget(myInterpolatingBox, 8, 2);
	layoutDock->addWidget(myToolTipBox, 9, 2);
	layoutDock->addWidget(mySnapOnHoverBox, 8, 3, 1, 2, Qt::AlignLeft);

	layoutDock->addWidget(myChartPropertyLabel, 10, 1);
	layoutDock->addWidget(myPointVisibilityBox, 11, 2);
	layoutDock->addWidget(myLineWidthBox, 12, 2);
	layoutDock->addWidget(myLineWidthLabel, 12, 3);
	layoutDock->addWidget(myLineWidthText, 12, 4);
	layoutDock->addWidget(mySetColorBox, 13, 2);
	layoutDock->addWidget(myPickColorButton, 13, 3);
	layoutDock->addWidget(myColorText, 13, 4);

	layoutDock->addWidget(myViewLabel, 14, 1);
	layoutDock->addWidget(myResetZoomButton, 15, 2, 1, 2, Qt::AlignLeft);

	QVBoxLayout *layoutDockVBox = new QVBoxLayout(this);
	layoutDockVBox->addLayout(layoutDock);
	layoutDockVBox->addStretch(1);
	QWidget* tempDock = new QWidget(this);
	tempDock->setLayout(layoutDockVBox);

	myChartPropertiesDock->setWidget(tempDock);
	myChartPropertiesDock->show();

	addDockWidget(Qt::RightDockWidgetArea, myChartPropertiesDock);
}

std::vector<int> SignalDataVisualizer::getSelection() {
	myPickedNodes.clear();
	myPickedNodes.push_back(2);

	return myPickedNodes;
}

void SignalDataVisualizer::setSelection(std::vector<int> _selected){
	foreach(int item , _selected){
		myPickedNodes.push_back(item);
    }
	if (myPickerButton->isChecked()) {
		myNodePickText->setText(QString::fromStdString(std::to_string(myHMesh->getNodeLabels().at(myPickedNodes.at(myPickedNodes.size()-1)))));
	}
}

void SignalDataVisualizer::updateList() {
	myPickTree->clear();
	if (myNodePickRadio->isChecked()) {
		QTreeWidgetItem* R1;
		R1 = addRootToTree(myPickTree, "Node " + myNodePickText->text(), false);
		addChildToTree(R1, "Ux_Re", false);
		addChildToTree(R1, "Uy_Re", false);
		addChildToTree(R1, "Uz_Re", false);
		addChildToTree(R1, "Ux_Im", false);
		addChildToTree(R1, "Uy_Im", false);
		addChildToTree(R1, "Uz_Im", false);
		myPickTree->expandAll();

		int nodeIndex = myHMesh->convertNodeLabelToNodeIndex(std::stoi(myNodePickText->text().toStdString()));
		
		std::cout << "\n-----Selected Node Displacement-----\n";
		std::cout << "---- Node " << myNodePickText->text().toStdString() << " @ FREQ "<< myHMesh->myOutputDatabase->getTimeDescription(myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX)[myVisualizerSetting->PROPERTY_CURRENT_TIMESTEP_INDEX] << "Hz ----\n";
		std::cout << std::showpos << "\tReal x: " << myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX].getResultScalarFieldAtNodes(STACCATO_x_Re, myVisualizerSetting->PROPERTY_CURRENT_TIMESTEP_INDEX)[nodeIndex] << std::endl;
		std::cout << "\tReal y: " << myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX].getResultScalarFieldAtNodes(STACCATO_y_Re, myVisualizerSetting->PROPERTY_CURRENT_TIMESTEP_INDEX)[nodeIndex] << std::endl;
		std::cout << "\tReal z: " <<myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX].getResultScalarFieldAtNodes(STACCATO_z_Re, myVisualizerSetting->PROPERTY_CURRENT_TIMESTEP_INDEX)[nodeIndex] << std::endl;
		std::cout << "\tMagni.: " <<myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX].getResultScalarFieldAtNodes(STACCATO_Magnitude_Re, myVisualizerSetting->PROPERTY_CURRENT_TIMESTEP_INDEX)[nodeIndex] << std::endl;
		std::cout << "------------------------------------\n";
		std::cout << "\tImag x: " <<myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX].getResultScalarFieldAtNodes(STACCATO_x_Im, myVisualizerSetting->PROPERTY_CURRENT_TIMESTEP_INDEX)[nodeIndex] << std::endl;
		std::cout << "\tImag y: " <<myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX].getResultScalarFieldAtNodes(STACCATO_y_Im, myVisualizerSetting->PROPERTY_CURRENT_TIMESTEP_INDEX)[nodeIndex] << std::endl;
		std::cout << "\tImag z: " <<myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX].getResultScalarFieldAtNodes(STACCATO_z_Im, myVisualizerSetting->PROPERTY_CURRENT_TIMESTEP_INDEX)[nodeIndex] << std::endl;
		std::cout << "\tMagni.: " <<myHMesh->myOutputDatabase->getVectorFieldFromDatabase()[myVisualizerSetting->PROPERTY_CURRENT_ANALYSIS_INDEX].getResultScalarFieldAtNodes(STACCATO_Magnitude_Im, myVisualizerSetting->PROPERTY_CURRENT_TIMESTEP_INDEX)[nodeIndex] << std::endl;
		std::cout << std::noshowpos << "------------------------------------\n";
	}
	else if (myElementPickRadio->isChecked()) {

	}

}

void SignalDataVisualizer::updateOutputTree() {
	if (myNodePickRadio->isChecked()) {
		QList<QTreeWidgetItem*> List = myPickTree->selectedItems();
		foreach (QTreeWidgetItem* item , List){
			addChildToTree(myHistoryRoot, item->text(0) + " at Node " + myNodePickText->text(), false);
			if (item->text(0) == "Ux_Re")
				add2dPlotToChart(std::stoi(myNodePickText->text().toStdString()), STACCATO_x_Re);
			else if (item->text(0) == "Uy_Re")
				add2dPlotToChart(std::stoi(myNodePickText->text().toStdString()), STACCATO_y_Re);
			else if (item->text(0) == "Uz_Re")
				add2dPlotToChart(std::stoi(myNodePickText->text().toStdString()), STACCATO_z_Re);
			else if (item->text(0) == "Ux_Im")
				add2dPlotToChart(std::stoi(myNodePickText->text().toStdString()), STACCATO_x_Im);
			else if (item->text(0) == "Uy_Im")
				add2dPlotToChart(std::stoi(myNodePickText->text().toStdString()), STACCATO_y_Im);
			else if (item->text(0) == "Uz_Im")
				add2dPlotToChart(std::stoi(myNodePickText->text().toStdString()), STACCATO_z_Im);
		}
		myPickTree->expandAll();
	}
	else if (myElementPickRadio->isChecked()) {

	}
}

void SignalDataVisualizer::myChartRangeUpdate() {
	myMinX->clearFocus();
	myMinY->clearFocus();
	myMaxX->clearFocus();
	myMaxY->clearFocus();
	myAxisX->setRange(myMinX->text().toDouble(), myMaxX->text().toDouble());
	myAxisY->setRange(myMinY->text().toDouble(), myMaxY->text().toDouble());
	myAxisLogX->setRange(myMinX->text().toDouble(), myMaxX->text().toDouble());
}

void SignalDataVisualizer::myClearPointSelection(){
	mySeries2scatter->clear();
	mySnapSeries2scatter->clear();

	myXCoordInfo->setText(tr(""));
	myYCoordInfo->setText(tr(""));

	foreach (ChartViewToolTip *callout , myCallOuts) {
		callout->scene()->removeItem(callout);
		myCallOuts.pop_back();
	}
	foreach (ChartViewToolTip *callout , mySnapCallOuts) {
		callout->scene()->removeItem(callout);
		mySnapCallOuts.pop_back();
	}
}

void SignalDataVisualizer::updateSeriesProperty() {
	foreach (QLineSeries* series, mySeriesList) {
		if (myPointVisibilityBox->isChecked()) {
			series->setPointsVisible(true);
		}
		else {
			series->setPointsVisible(false);
		}
		QPen pen(series->color());
		pen.setWidth(myLineWidthText->text().toDouble());
		series->setPen(pen);
	}
}

void SignalDataVisualizer::setLineWidthWidgets() {
	myLineWidthLabel->setEnabled(myLineWidthBox->isChecked());
	myLineWidthText->setEnabled(myLineWidthBox->isChecked());
	if (!myLineWidthBox->isChecked()) {
		myLineWidthText->setText(tr("2.0"));
		updateSeriesProperty();
	}
}

void SignalDataVisualizer::updateToolTip() {
	if (!myToolTipBox->isChecked()) {
		foreach (ChartViewToolTip *callout, myCallOuts) {
			callout->scene()->removeItem(callout);
			myCallOuts.pop_back();
		}
	}
}

void SignalDataVisualizer::deleteSeries(){
	QList<QTreeWidgetItem*> List = myOutputTree->selectedItems();
	foreach (QTreeWidgetItem* item , List) {
		int currIndex = myOutputTree->currentIndex().row();
		myChart2D->removeSeries(mySeriesList.at(currIndex));
		if (myOrdinateList.size() != 0) {
			myChart2D->removeAxis(myOrdinateList.last());
			myOrdinateList.removeAt(currIndex);
		}
		mySeriesList.removeAt(currIndex);
		myHistoryRoot->removeChild(item);
	}
}

void SignalDataVisualizer::addOrdinateToSeries() {
	QList<QTreeWidgetItem*> List = myOutputTree->selectedItems();
	foreach (QTreeWidgetItem* item , List) {
		int currIndex = myOutputTree->currentIndex().row();
		QValueAxis* newAxisY = new QValueAxis();

		newAxisY->setTitleText(item->text(0));
		newAxisY->setLabelFormat("%d");

		myOrdinateList.append(newAxisY);

		myChart2D->addAxis(myOrdinateList.last(), Qt::AlignLeft);

		mySeriesList.at(currIndex)->attachAxis(myOrdinateList.last());
	}
}

void SignalDataVisualizer::resetChart() {
	for (int i = mySeriesList.size()-1; i >=0 ; i--) {
		myChart2D->removeSeries(mySeriesList.at(i));
		mySeriesList.removeAt(i);
	}
	myHistoryRoot->takeChildren();
	myClearPointSelection();
	myPickTree->clear();
	myNodePickText->setText("");
}

void SignalDataVisualizer::saveChart() {
	QFileDialog* saveWindow = new QFileDialog(this);
	QString fileName = QFileDialog::getSaveFileName(this, "Save Image", "STACCATO_Chart", ".png");

	QPixmap imageHandle = chartView2D->grab();
	imageHandle.save(fileName + ".png", "PNG", -1);
}

void SignalDataVisualizer::myLogScaleTriggered() {
	if (myLogXAxisBox->isChecked()) {
		myChart2D->removeAxis(myAxisX);
		myChart2D->addAxis(myAxisLogX, Qt::AlignBottom);
		for (int i = mySeriesList.size() - 1; i >= 0; i--) {
			mySeriesList.at(i)->detachAxis(myAxisX);
			mySeriesList.at(i)->attachAxis(myAxisLogX);
		}
		mySeries2scatter->attachAxis(myAxisLogX);
		mySnapSeries2scatter->attachAxis(myAxisLogX);
	}
	else {
		myChart2D->removeAxis(myAxisLogX);
		myChart2D->addAxis(myAxisX, Qt::AlignBottom);
		for (int i = mySeriesList.size() - 1; i >= 0; i--) {
			mySeriesList.at(i)->detachAxis(myAxisLogX);
			mySeriesList.at(i)->attachAxis(myAxisX);
		}
		mySeries2scatter->attachAxis(myAxisX);
		mySnapSeries2scatter->attachAxis(myAxisX);
		autoScale(myValueMinY, myValueMaxY);
	}
	updateAxesConnections();
}

void SignalDataVisualizer::myColorSelectionTriggered() {
	myPickColorButton->setEnabled(mySetColorBox->isChecked());
	myColorText->setEnabled(mySetColorBox->isCheckable());
}

void SignalDataVisualizer::myColorPickWidget() {
	QColorDialog* colorPickDialog = new QColorDialog(this);
	QColor color = colorPickDialog->getColor(Qt::yellow, this);
	if (color.isValid()) {
		myColorText->setText(color.name());
		QList<QTreeWidgetItem*> List = myOutputTree->selectedItems();

		foreach (QTreeWidgetItem* item , List) {
			int currIndex = myOutputTree->currentIndex().row();

			QPen pen(color);
			pen.setWidth(myLineWidthText->text().toDouble());
			mySeriesList.at(currIndex)->setPen(pen);
		}
	}
}

void SignalDataVisualizer::myResetChart() {
	myChart2D->zoomReset();
}

void SignalDataVisualizer::updateSnap() {
	if(!mySnapOnHoverBox->isChecked())
		mySnapSeries2scatter->clear();
}

void SignalDataVisualizer::connectVisualizerSetting(VisualizerSetting* _setting) {
	myVisualizerSetting = _setting;
}
