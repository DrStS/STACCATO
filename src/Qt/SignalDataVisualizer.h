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
/***********************************************************************************************//**
* \file SignalDataVisualizer.h
* This file holds the class of SignalDataVisualizer.
* \date 2/20/2018
**************************************************************************************************/
#ifndef SIGNALDATAVISUALIZER_H
#define SIGNALDATAVISUALIZER_H

#include "HMeshToVtkUnstructuredGrid.h"
#include <STACCATO_Enum.h>
#include <DiscreteVisualizer.h>
#include "FieldDataVisualizer.h"
#include "observer.h"

// QT5
#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QLineEdit>
#include <QComboBox>
#include <QtCharts/QChartView>
#include <QScatterSeries>
#include <QLineSeries>
#include <QValueAxis>
#include <QLogValueAxis>

QT_CHARTS_USE_NAMESPACE

// OCC
#include <AIS_InteractiveContext.hxx>
// SimuliaOBD
#include "SimuliaODB.h"
// Chart
#include "chartview.h"

// forward declaration
class ChartViewToolTip;
class OccViewer;
class QTextEdit;
class QCheckBox;
class QGroupBox;
class QSpinBox;
class QFormLayout;
class QTreeWidget;
class QRadioButton;
class QTreeWidgetItem;
class QPointF;
class QGestureEvent;

class HMesh;

namespace Ui {
	class SignalDataVisualizer;
}
/********//**
* \brief Class SignalDataVisualizer the core of the GUI
***********/
class SignalDataVisualizer : public QMainWindow, public DiscreteVisualizer, public Observer {
	Q_OBJECT

public:
	/***********************************************************************************************
	* \brief Constructor
	* \author Stefan Sicklinger
	***********/
	explicit SignalDataVisualizer();
	/***********************************************************************************************
	* \brief Destructor
	* \author Stefan Sicklinger
	***********/
	~SignalDataVisualizer();
	/***********************************************************************************************
	* \brief Initiate the SignalDataVisualizer
	* \author Stefan Sicklinger
	***********/
	void initiate(void);
	/***********************************************************************************************
	* \brief Set Subject for the current Observer
	* \author Harikrishnan Sreekumar
	***********/
	void attachSubject(FieldDataVisualizer* _fieldDataVisualizer);
	/***********************************************************************************************
	* \brief Interactive Update
	* \author Harikrishnan Sreekumar
	***********/
	void update(void);

protected:
	/***********************************************************************************************
	* \brief Create all Actions of Qt
	* \author Stefan Sicklinger
	***********/
	void createActions(void);
	void createMenus(void);
	void createDockWindow(void);
	void createChart(void);
	QTreeWidgetItem* addRootToTree(QTreeWidget*, QString, bool);
	void addChildToTree(QTreeWidgetItem*, QString, bool);
	std::vector<int> getSelection(void);
	void enableInteractiveClick(QLineSeries*);
	void add2dPlotToChart(int _nodeLabel, STACCATO_VectorField_components _type);
	void autoScale(double, double);
	void updateLegends(void);
	void updateAxesConnections(void);
private slots:
	void updateList(void);
	void updateOutputTree(void);
	void tooltip(QPointF _point, bool _state);
	void handleClickedPoint(const QPointF &point);
	void myChartRangeUpdate(void);
	void myClearPointSelection(void);
	void updateSeriesProperty(void);
	void setLineWidthWidgets(void);
	void updateToolTip(void);
	void prepareTreeMenu(const QPoint& _pos);
	void addOrdinateToSeries(void);
	void deleteSeries(void);
	void resetChart(void);
	void saveChart(void);
	void myLogScaleTriggered(void);
	void myColorSelectionTriggered(void);
	void myColorPickWidget(void);
	void myRangeWidgetXUpdate(qreal, qreal);
	void myRangeWidgetYUpdate(qreal, qreal);
	void myResetChart(void);
	void updateSnap(void);
public:
	void setSelection(std::vector<int>);
	QPushButton* myPickerButton;
private:
	/// File action.
	QAction* myExitAction;
	/// Help Action
	QAction* myAboutAction;
	/// Menus.
	QMenu* myFileMenu;
	QMenu* myHelpMenu;

	/// Dockets
	QDockWidget *myOutputTreeDock;
	QDockWidget *myChartPropertiesDock;

	/// Tree Dock Widget
	QLineEdit* myNodePickText;
	QLabel* myNodePickLabel;

	/// Axis Range Labels
	QLabel* myMinXLabel;
	QLabel* myMaxXLabel;
	QLabel* myMinYLabel;
	QLabel* myMaxYLabel;
	QLabel* myRangeLabel;
	QLabel* mySelectionLabel;
	QLabel* myInterpLabel;
	QLabel* myChartPropertyLabel;
	QLabel* myXCoordInfoLabel;
	QLabel *myYCoordInfoLabel;
	QLabel *myLineWidthLabel;
	QLabel *myViewLabel;

	/// Axis Range LineEdits
	QLineEdit* myMinX;
	QLineEdit* myMaxX;
	QLineEdit* myMinY;
	QLineEdit* myMaxY;
	QLineEdit* myXCoordInfo;
	QLineEdit *myYCoordInfo;
	QLineEdit* myLineWidthText;
	QLineEdit* myColorText;

	/// Check Boxes
	QCheckBox* myInterpolatingBox;
	QCheckBox* myToolTipBox;
	QCheckBox* myPointVisibilityBox;
	QCheckBox* myLineWidthBox;
	QCheckBox* myLogXAxisBox;
	QCheckBox* myLogYAxisBox;
	QCheckBox* mySetColorBox;
	QCheckBox* mySnapOnHoverBox;

	/// Radio Buttons
	QRadioButton* myNodePickRadio;
	QRadioButton* myElementPickRadio;
	QRadioButton* myForceNodeRadio;
	QRadioButton* myForceIntPointsRadio;

	/// Push Buttons
	QPushButton* myPickListButton;
	QPushButton* myPickAddButton;
	QPushButton* mySaveButton;
	QPushButton* myResetButton;
	QPushButton* myPickColorButton;
	QPushButton *myResetZoomButton;
	QPushButton* myClearSelectionButton;

	/// Tree Widgets
	QTreeWidget* myOutputTree;
	QTreeWidget* myPickTree;
	QTreeWidgetItem* myHistoryRoot;

	QGraphicsSimpleTextItem *myCoordX;
	QGraphicsSimpleTextItem *myCoordY;

	/// Tooltips
	ChartViewToolTip * myToolTip;
	QList<ChartViewToolTip *> myCallOuts;
	QList<ChartViewToolTip *> mySnapCallOuts;

	/// Lists
	QList<QLineSeries *> mySeriesList;
	QList<QValueAxis *> myOrdinateList;

	/// Plot Series
	QLineSeries* mySeries2D;
	QScatterSeries* mySeries2scatter;
	QScatterSeries* mySnapSeries2scatter;

	/// Chart
	QChart* myChart2D;
	ChartView *chartView2D;
	
	/// Plot Data
	std::vector<double> yValues;// Displacement at node
	std::vector<int> xValues;	// Frequency Range in Integer
	double myValueMinY;			// Range
	double myValueMaxY;			// Range

	/// Menu Actions
	QAction* myAddAxisAction;
	QAction* myDeleteAction;

	/// Axes
	QLogValueAxis* myAxisLogX;
	QValueAxis *myAxisX;
	QValueAxis *myAxisY;

	/// Global Flag
	int hoverFlag = 0;
	QPointF mySnap;

	/// Picking
	std::vector<int> myPickedNodes;

	FieldDataVisualizer* myFieldDataVisualizerSubject;
};

#endif /* SIGNALDATAVISUALIZER_H */
