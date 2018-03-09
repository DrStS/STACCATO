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
#include <VisualizerSetting.h>

VisualizerSetting::VisualizerSetting()
{
	// Initialize default property
	PROPERTY_SCALARBAR_TITLE = "";
	PROPERTY_SCALARBAR_VISIBILITY = true;		// Scalar Bar is enabled by Default
	PROPERTY_FRAMEID = 0;						// Set the initial frame as Zero
	PROPERTY_FIELD_TYPE = STACCATO_x_Re;
	PROPERTY_SCALING_FACTOR = 1;

	PROPERTY_ANIMATION_DURATION = 5;
	PROPERTY_ANIMATION_REPEAT   = 1;

	PROPERTY_RESULTS_AVALABLE = false;

	myViewModeLabelMap["Surface"] = STACCATO_FieldProperty_Surface;
	myViewModeLabelMap["Surface with Edges"] = STACCATO_FieldProperty_SurfaceWithEdges;
	myViewModeLabelMap["Wireframe"] = STACCATO_FieldProperty_Wireframe;
}

VisualizerSetting::~VisualizerSetting()
{
}

void VisualizerSetting::setCommuniationToFieldDataVisualizer(FieldDataVisualizer& _fieldDataVisualizer) {
	myFieldDataVisualizer = &_fieldDataVisualizer;
	myFieldDataVisualizer->connectVisualizerSetting(this);
	std::cout << ">> Connection to FieldDataVisualizer is set by VisualizerSetting.\n";
}

void VisualizerSetting::commitViewSetting(STACCATO_FieldProperty_type _property) {

	switch (_property)
	{
	case STACCATO_FieldProperty_Surface:
		myFieldDataSetting = new SurfaceSetting();
		std::cout << ">> Visualizer Setting set as: Surface.\n";
		break;
	case STACCATO_FieldProperty_SurfaceWithEdges:
		myFieldDataSetting = new SurfaceWithEdgesSetting();
		std::cout << ">> Visualizer Setting set as: SurfaceWithEdges.\n";
		break;
	case STACCATO_FieldProperty_Wireframe:
		myFieldDataSetting = new WireFrameSetting();
		std::cout << ">> Visualizer Setting set as: Wireframe.\n";
		break;
	default:
		std::cerr << "Unidentified VisualizerSetting.\n";
		break;
	}
	
	updateSetting();
}

void VisualizerSetting::commitVectorFieldComponent(STACCATO_VectorField_components _component) {
	PROPERTY_FIELD_TYPE = _component;
}

void VisualizerSetting::updateSetting() {
	myFieldDataVisualizer->myHMeshToVtkUnstructuredGridSetScalar(PROPERTY_FIELD_TYPE, PROPERTY_FRAMEID);
	myFieldDataVisualizer->myHMeshToVtkUnstructuredGridSetVector(PROPERTY_FRAMEID);
	myFieldDataVisualizer->plotVectorField();																	// Update Plot
}

void VisualizerSetting::commitScalarBar(bool _enable) {
	PROPERTY_SCALARBAR_VISIBILITY = _enable;
}

void VisualizerSetting::commitScalingFactor(double _scalingFactor) {
	PROPERTY_SCALING_FACTOR = _scalingFactor;
}

void VisualizerSetting::commitCurrentFrame(int _frameID) {
	std::cout << ">> Committed to Frame: " << _frameID << std::endl;
	PROPERTY_FRAMEID = _frameID;
}

void VisualizerSetting::setScalarbarTitle(std::string _title) {
	PROPERTY_SCALARBAR_TITLE = new char[_title.size() + 1];
	strcpy(PROPERTY_SCALARBAR_TITLE, _title.c_str());
}

void VisualizerSetting::generateAnimation() {
	myFieldDataVisualizer->animate(PROPERTY_FIELD_TYPE);
	myFieldDataVisualizer->plotVectorFieldAtIndex(PROPERTY_FRAMEID);
}

void VisualizerSetting::playAnimation() {
	myFieldDataVisualizer->myAnimationScenePlayProc(PROPERTY_ANIMATION_DURATION, PROPERTY_ANIMATION_REPEAT);
}

void VisualizerSetting::stopAnimation() {
	myFieldDataVisualizer->myAnimationSceneStopProc();
}

void VisualizerSetting::visualizeAnimationFrames(int _duration, int _repeat) {
	PROPERTY_ANIMATION_DURATION = _duration;
	PROPERTY_ANIMATION_REPEAT = _repeat;

	playAnimation();
}

void VisualizerSetting::setResultAvailable(bool _available) {
	PROPERTY_RESULTS_AVALABLE = _available;
}