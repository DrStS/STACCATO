// In this example, we read the contents of an output database and dump them
// stdout.
//
//


//
// System includes
//
#include <stdlib.h>
#include <stdio.h>
#if (defined(HP) && (! defined(HKS_HPUXI)))
#include <iostream.h>
#include <iomanip.h>
#else
#include <iostream>
#include <iomanip>
using namespace std;
#endif

// Begin local includes
#include <odb_API.h>
#include <odb_Coupling.h>
#include <odb_MPC.h>
#include <odb_ShellSolidCoupling.h>
// End local includes



// Declare local function prototypes
int odbDump(int argc, char **argv);

void odbE_printFullOdb          (odb_Odb&, bool printPath);
void odbE_printParts            (odb_PartRepository& );
void odbE_printPart             (odb_Part& );
void odbE_printInstanceCon      (odb_InstanceRepository& );
void odbE_printInstance         (odb_Instance& );
void odbE_printAssembly         (odb_Assembly& );
void odbE_printDatumCsys        (const odb_DatumCsys& myDC, 
				 odb_String space="");
void odbE_printSets             (const odb_SetRepository& nSets,
                                 odb_String space="");
void odbE_printSet              (odb_Set& set,
                                 odb_String space="");
void odbE_printNodes            (const odb_SequenceNode& n,
                                 odb_String space="");
void odbE_printNode             (const odb_Node& n,
                                 odb_String space="");
void odbE_printElements         (const odb_SequenceElement& e,
                                 odb_String space="");
void odbE_printElement          (const odb_Element& e,
                                 odb_String space="");
void odbE_printInteractionCon   (const odb_InteractionRepository&,
				 odb_Odb& odb);
void odbE_printInteractions     (const odb_Interaction&,
				 odb_Odb& odb,
                                 odb_String space="");
void odbE_printInteractionProperty (const odb_InteractionProperty&,
                                 odb_String space="");
void odbE_printSSC_Std          (const odb_SurfaceToSurfaceContactStd& ssc,
				 odb_Odb& odb,
                                 odb_String space="");
void odbE_printSSC_Exp          (const odb_SurfaceToSurfaceContactExp& ssc,
				 odb_Odb& odb,
                                 odb_String space="");

void odbE_printConstraintCon    (const odb_ConstraintRepository& CR, 
				 odb_Odb& odb);

void odbE_printTie              (const odb_Tie& tie,
				 odb_Odb& odb,
                                 odb_String space="");
void odbE_printDisplayBody      (const odb_DisplayBody& db,
				 odb_Odb& odb,
                                 odb_String space="");
void odbE_printCoupling         (const odb_Coupling& coup,
				 odb_Odb& odb,
                                 odb_String space="");
void odbE_printMPC              (const odb_MPC& mpc,
				 odb_Odb& odb,
                                 odb_String space="");
void odbE_printShellSolidCoupling(const odb_ShellSolidCoupling& coup,
				 odb_Odb& odb,
                                 odb_String space="");

void odbE_printStep             (odb_Step& s,
                                 odb_String space="",
                                 int odbE_printLevel=0);
void odbE_printFrame            (odb_Frame& f,
                                 odb_String space="",
                                 int odbE_printLevel=0);
void odbE_printField            (odb_FieldOutput& ,
                                 odb_String space="",
                                 int odbE_printLevel=0);
void odbE_printFieldValueCon    (const odb_SequenceFieldValue& ,
                                 const odb_SequenceInvariant& invars,
                                 odb_String space="",
				 bool complexData=false);
void odbE_printFieldBulkDataCon (const odb_SequenceFieldBulkData& fvCon,
				 const odb_SequenceInvariant& invars,
				 odb_String space="",
				 bool complexData=false);
void odbE_printBulkField       (const odb_FieldBulkData& bulkData,
				 const odb_SequenceInvariant& invars,
				 odb_String space="",
				 bool complexData=false);
void odbE_printFieldLocationCon (odb_SequenceFieldLocation ,
			         odb_String space="");
void odbE_printFieldValue       (const odb_FieldValue& ,
                                 const odb_SequenceInvariant& invars,
                                 odb_String space="",
				 bool complexData=false);
void odbE_printFieldLocation    (const odb_FieldLocation& ,
                                 odb_String space="");
void odbE_printHistoryRegion    (odb_HistoryRegion& h,
                                 odb_String space="",
                                 int odbE_printLevel=0);
void odbE_printHistoryOutput    (const odb_HistoryOutput& ,
                                 odb_String space="");
void odbE_printHistoryPoint     (const odb_HistoryPoint& ,
                                 odb_String space="");
void odbE_printFaces            (const odb_SequenceElementFace faces);
void odbE_printSectionCatCon    (const odb_SectionCategoryRepository& catCon);
void odbE_printSectionCat       (const odb_SectionCategory& cat);



int ABQmain(int argc, char **argv)
{
    odb_String path = "odbAccess.odb";
    if (argc > 1)
	path = argv[1];
    bool printPath = true;
    if (argc > 2) {
	odb_String noPath(argv[2]);
	if (noPath == "-nopath")
	    printPath = false;
    }    
    cout << "file " << path.CStr() << endl << endl << endl;    
	odb_Odb& odb = openOdb("C:\\software\\libs\\SIMULIA\\viewer_tutorial.odb");
    odbE_printFullOdb(odb, printPath);    
    cout << "end of output" << endl;
    return (0);
}

void odbE_printFullOdb(odb_Odb& odb, bool printPath)
{
    cout << "__________ ODB '";
    cout << odb.name().CStr() << " '__________" << endl;
    cout << "analysisTitle '" << odb.analysisTitle().CStr() << "'" << endl;
    cout << "description '" <<  odb.description().CStr() << "'" << endl;
    if (printPath)
	cout << "path '" << odb.path().CStr() << "'" << endl;        
    if (odb.hasSectorDefinition())
    {
	odb_SectorDefinition sd = odb.sectorDefinition();
	cout << "Number of Sectors: "<< sd.numSectors()<<endl;
	odb_SequenceSequenceFloat symAx = sd.symmetryAxis();
	cout << "Symmetry Axis:"<<endl;
	cout <<"  Start Point: ["; 
	cout <<symAx[0][0]<<","<<symAx[0][1]<<","<<symAx[0][2]<<"]"<<endl;
	cout <<"   End  Point: ["; 
	cout <<symAx[1][0]<<","<<symAx[1][1]<<","<<symAx[1][2]<<"]"<<endl;        
    }

    odb_PartRepository& parts = odb.parts();
    odbE_printParts (parts);

    odbE_printSectionCatCon (odb.sectionCategories());


    odb_InstanceRepository iCon = odb.rootAssembly().instances();
    odbE_printInstanceCon (iCon);

    odbE_printAssembly(odb.rootAssembly());

    odb_InteractionRepository interCon = odb.interactions();
    if(interCon.size()>0)
      odbE_printInteractionCon(interCon,odb);

    odb_ConstraintRepository constCon = odb.constraints();
    if(constCon.size()>0)
      odbE_printConstraintCon(constCon,odb);

    odb_StepRepository sCon = odb.steps();
    odb_StepRepositoryIT sIter (sCon);
    for (sIter.first(); !sIter.isDone(); sIter.next()) 
    {
        odbE_printStep(sCon[sIter.currentKey()], "    ", 9999);        
    }
}

void odbE_printParts(odb_PartRepository& PC )
{
    cout << endl;
    cout << endl;
    cout << "__________ PART CONTAINER __________" << endl;
    cout << "    number of parts : " << PC.size() << endl;
    odb_PartRepositoryIT iter(PC);    
    for (iter.first(); !iter.isDone(); iter.next()) 
    {
        odb_Part& part = PC[iter.currentKey()];
        cout << "    ----- odb_Part key : ";
        cout << iter.currentKey().CStr();
        cout << " ---------" << endl;
        odbE_printPart (part);
    }
    cout << endl << endl;
}

void odbE_printPart(odb_Part& p )
{
    cout << "    ______ PART ____ '" << endl;
    cout << "    Part name : " <<  p.name().CStr() << " embeddedSpace : "
         <<  p.embeddedSpace() << endl;
    const odb_SequenceNode& nodes = p.nodes();
    odbE_printNodes (nodes, "        ");
    odb_SequenceElement elements = p.elements();
    odbE_printElements (elements, "        ");
    cout << endl << "    ____ NodeSetContainer ____ " << endl;
    odbE_printSets (p.nodeSets(), "    ");
    cout << endl << "    ____ ElementSetContainer ____ " << endl;
    odbE_printSets (p.elementSets(), "    ");
    cout << endl << "    ____ SurfaceContainer ____ " << endl;
    odbE_printSets (p.surfaces(), "    ");
    cout << endl << endl;
}

void odbE_printInstanceCon(odb_InstanceRepository& iC )
{
    cout << endl << endl;
    cout << "__________ INSTANCE CONTAINER __________" << endl;
    cout << "    number of instances : " << iC.size() << endl;
    odb_InstanceRepositoryIT iter(iC);
    for (iter.first(); !iter.isDone(); iter.next()) 
    {
        odb_Instance& inst = iC[iter.currentKey()];
        odbE_printInstance (inst);
    }
    cout << endl << endl;
}

void odbE_printInstance(odb_Instance& i)
{
    cout << endl << endl;
    cout << "    _____ Instance '" << i.name().CStr();
    cout << "' _________" << endl;
    cout << "      embeddedSpace : " <<  i.embeddedSpace() << endl;
    odb_SequenceNode& nodes = i.nodes();
    odbE_printNodes (nodes, "        ");
    odb_SequenceElement& elements = i.elements();
    odbE_printElements (elements, "        ");
    cout << endl << "    ____ NodeSetContainer ____ " << endl;
    odbE_printSets (i.nodeSets(), "    ");
    cout << endl << "    ____ ElementSetContainer ____ " << endl;
    odbE_printSets (i.elementSets(), "    ");
    cout << endl << "    ____ SurfaceContainer ____ " << endl;
    odbE_printSets (i.surfaces(), "    ");
    cout << endl << endl;
}

void odbE_printAssembly(odb_Assembly& a)
{
    cout << "__________ ASSEMBLY '" <<  a.name().CStr() << "' __________" << endl;
    cout << "    embeddedSpace : " <<  a.embeddedSpace() << endl;
    const odb_SequenceNode& nodes = a.nodes();
    odbE_printNodes (nodes, "        ");
    odb_SequenceElement elements = a.elements();
    odbE_printElements (elements, "        ");
    cout << endl;
    cout << endl << "    ____ NodeSetContainer ____ " << endl;
    odbE_printSets (a.nodeSets(), "    ");
    cout << endl << "    ____ ElementSetContainer ____ " << endl;
    odbE_printSets (a.elementSets(), "    ");
    cout << endl << "    ____ SurfaceContainer ____ " << endl;
    odbE_printSets (a.surfaces(), "    ");
    cout << endl << endl;
}

void odbE_printDatumCsys(const odb_DatumCsys& myDC, odb_String space)
{
    cout << space.CStr() << "__Datum Csys__" << endl;
    cout << space.CStr() << "Name: " << myDC.name().CStr() << endl;
    cout << space.CStr() << "Type: " << myDC.type() << endl;
    const float* x_axis = myDC.xAxis();
    const float* y_axis = myDC.yAxis();
    const float* z_axis = myDC.zAxis();
    const float* origin = myDC.origin();
    cout << space.CStr() << "Origin: " << origin[0] << ", " << origin[1]//
         << ", " << origin[2] << endl;
    cout << space.CStr() << "Xaxis: " << x_axis[0] << ", " << x_axis[1]//
         << ", " << x_axis[2] << endl;
    cout << space.CStr() << "Yaxis: " << y_axis[0] << ", " << y_axis[1]//
         << ", " << y_axis[2] << endl;
    cout << space.CStr() << "Zaxis: " << z_axis[0] << ", " << z_axis[1]//
         << ", " << z_axis[2] << endl;
}

void odbE_printSets( const odb_SetRepository& sets, odb_String space)
{
    odb_SetRepositoryIT iter(sets);
    for (iter.first(); !iter.isDone(); iter.next()) {
        odb_Set s = iter.currentValue();
        odbE_printSet (s, space+"    ");
    }
}

void odbE_printSet(odb_Set& set, odb_String space)
{
    if (set.name().empty())
    {
        cout << space.CStr() << "The set is empty." << endl;
        return;
    }

    cout << endl;
    int setSize = set.size();
    switch(set.type()) 
    {
    case odb_Enum::NODE_SET:
        cout << space.CStr() << "Node ";
        break;
    case odb_Enum::ELEMENT_SET:
        cout << space.CStr() << "Element ";
        break;
    case odb_Enum::SURFACE_SET:
        cout << space.CStr() << "Surface ";
        break;
    }
    cout << "Set '" << set.name().CStr() << "'" << endl; 
    cout << space.CStr() << "    Set Size : " << setSize << endl;
    odb_SequenceString names = set.instanceNames();
    int numInstances = names.size();
    cout << space.CStr() <<  "    number of instances in set: " 
         << numInstances << endl;
    int i; // for loop index    
    for (i=0; i<numInstances; i++) 
        cout << space.CStr() << "        instances : " << 
            names.constGet(i).CStr() << endl;
    
    for (i=0; i<numInstances; i++) 
    {
        odb_String name = names.constGet(i);        
        cout << space.CStr() << "    Instance " << name.CStr() << endl;            
	switch(set.type()) 
        {
	case odb_Enum::NODE_SET:
        {
	    const odb_SequenceNode& nodesInMySet = set.nodes(name);
	    int n_max = nodesInMySet.size();
	    for (int n=0; n<n_max; n++) 
            {
		odbE_printNode(nodesInMySet.node(n),
			       space + "          "); 
            }	    
        }    
        break;                
	case odb_Enum::ELEMENT_SET:
        {	    
	    const odb_SequenceElement& elemsInMySet = set.elements(name);
	    int n_max = elemsInMySet.size();
	    for (int n=0; n<n_max; n++) 
            {
		odbE_printElement(elemsInMySet.element(n),
				  space + "          "); 
            }	    
	  }
        break;
	case odb_Enum::SURFACE_SET:
        {
	    bool hasFaces = true;
	    bool hasNodes = false;
	    bool hasElements = false;
	    const odb_SequenceElement& els = set.elements(name);
	    const odb_SequenceElementFace& faces = set.faces(name);
	    const odb_SequenceNode& nds = set.nodes(name);
	    int els_size = els.size();
	    int faces_size = faces.size();
	    int nds_size = nds.size();

	    if(els_size)
                hasElements = true;            
	    if(faces_size)
                hasFaces = true;
            if(nds_size)
                hasNodes = true;
	    
	    if(hasElements && hasFaces)
            {
		cout << endl << "    ____ Element & Face Data  ____ " << endl;
		for (int n=0; n<els_size; n++) 
                {
		    odbE_printElement(els.element(n),
				      space+"          "); 
		    cout << space.CStr() << "          face : " << faces.constGet(n) << endl;
                }
            }
	    else if(hasElements)
            {
		cout << endl << "    ____ Element Data  ____ " << endl;
		for (int n=0; n<els_size; n++) 
                    odbE_printElement(els.element(n), space+"          ");               
            }
	    else
            {
		cout << endl << "    ____ Nodal Data  ____ " << endl;
		for (int n=0; n<nds_size; n++) 
                    odbE_printNode(nds.node(n), space + "          ");                 
            }
	    
	    break;
        }
	default:
            cerr << "UNKNOWN SET TYPE" << endl;
            break;
	}
    }
    cout << endl;
}


void odbE_printFaces( const odb_SequenceElementFace faces)
{
    int numF = faces.size();
    for (int i=0; i<numF; i++) 
        cout <<  "                    " << faces.constGet(i) << endl;    
}

void odbE_printNodes( const odb_SequenceNode& nC, odb_String space)
{
    cout << space.CStr() << "number of nodes : " << nC.size() << endl;
    int numN = nC.size();
    for (int i=0; i<numN; i++) 
    {
        const odb_Node n = nC.node(i);
        odbE_printNode (n, space + "    ");
    }
}

void odbE_printElements( const odb_SequenceElement& eC, odb_String space)
{
    cout << space.CStr() << "number of elements : " << eC.size() << endl;
    int numEls = eC.numElements();   
    for (int i=0; i<numEls; i++) 
    {
        const odb_Element e = eC.element(i);
        odbE_printElement (e, space + "    ");
    }
}

void odbE_printNode( const odb_Node& n, odb_String space)
{
    const float * const coords = n.coordinates();
    char formattedOut[256];
    sprintf(formattedOut," %9d [%10.3f %10.3f %10.3f]\n",n.label(),
            coords[0],coords[1],coords[2]);
    cout << formattedOut;    
}

void odbE_printElement( const odb_Element& e, odb_String space)
{
    cout.setf(ios::right, ios::adjustfield); 
    cout << setw(9) <<  e.label() << " " << e.type().CStr() << " [";
    int elemConSize;
    const int* const conn = e.connectivity(elemConSize); 
    for (int j=0;j<elemConSize; j++)
        cout << setw(9) << conn[j];
    cout << " ] " << endl;
}

/////////////////////////////////////////////////
void odbE_printStep(odb_Step& step, odb_String space, int odbE_printLevel)
{
    cout << space.CStr() << "__________ STEP '";
    cout << space.CStr() << step.name().CStr() << "' _______________" << endl;
    cout << space.CStr() << "  description : " << step.description().CStr()  << endl;
    cout << space.CStr() << "  domain      : " << step.domain() << endl;
    cout << space.CStr() << "  procedure   : " << step.procedure().CStr() << endl;
    cout << space.CStr() << "  timePeriod  : " << step.timePeriod() << endl;
    cout << endl;
    
    if (odbE_printLevel>0) 
    {
        --odbE_printLevel;
        odb_SequenceFrame& fCon = step.frames();
        int numFrames = fCon.size();
        for (int f=0; f<numFrames; f++) 
        {
            odb_Frame frame = fCon.constGet(f);
            odbE_printFrame(frame, space+"    ", odbE_printLevel);
        }
        odb_HistoryRegionRepository hoCon = step.historyRegions();
        odb_HistoryRegionRepositoryIT hoIter (hoCon);
        for (hoIter.first(); !hoIter.isDone(); hoIter.next()) 
        {
            odb_HistoryRegion ho = hoIter.currentValue();
            odbE_printHistoryRegion (ho, space+"    ", odbE_printLevel);
        }
    }    
}


void odbE_printFrame(odb_Frame& frame, odb_String space, int odbE_printLevel)
{
    cout << space.CStr() << "__________FRAME_______________" << endl;
    cout << space.CStr() << "  description     : " << frame.description().CStr() << endl;
    cout << space.CStr() << "  incrementNumber : " << frame.incrementNumber()  << endl;
    cout << space.CStr() << "  frameValue      : " << frame.frameValue() << endl;
    cout << space.CStr() << "  domain          : " << frame.domain() << endl;
    cout << endl;
    if( odbE_printLevel>0 ) 
    {
        --odbE_printLevel;
        odb_FieldOutputRepository& fieldCon = frame.fieldOutputs();
        odb_FieldOutputRepositoryIT fieldConIT(fieldCon);
        for (fieldConIT.first(); !fieldConIT.isDone(); fieldConIT.next()) 
        {
            odb_FieldOutput& field = fieldCon[fieldConIT.currentKey()]; 
            odbE_printField(field, space+"    ", odbE_printLevel);
        }
    }
}

void odbE_printField(odb_FieldOutput& field, odb_String space, int odbE_printLevel)
{
    cout << space.CStr() << "__________FIELD '";
    cout << field.name().CStr() << "' _______________" << endl;
    cout << space.CStr() << "  description     : "
         << field.description().CStr() << endl;
    cout << space.CStr() << "  type            : "
         << field.type()  << endl;
    cout << space.CStr() << "  validInvariants : ";
    int numInvar = field.validInvariants().size();
    for (int i=0; i<numInvar; i++) 
        cout <<  field.validInvariants().constGet(i) << " "; 
    cout  <<  endl;
    cout << space.CStr() << "  componentLabels : ";
    odb_SequenceString availableComponents = field.componentLabels();
    int numComp = availableComponents.size();
    for (int j=0; j<numComp; j++) 
    {
        cout <<  availableComponents[j].CStr() << " ";
    }
    cout << endl;    
    bool isComplex = field.isComplex();
    if (odbE_printLevel>0) 
    {
        --odbE_printLevel;
        odb_SequenceFieldLocation flCon = field.locations();
        odbE_printFieldLocationCon (flCon, space + "    ");
	const odb_SequenceFieldValue& fvCon = field.values();
	if (numInvar)
            odbE_printFieldValueCon(fvCon,field.validInvariants(), space + "    ",isComplex);	
        const odb_SequenceFieldBulkData& bdbCon = field.bulkDataBlocks();	
        odbE_printFieldBulkDataCon(bdbCon,field.validInvariants(), space + "    ",isComplex);
    }
}

void odbE_printFieldBulkDataCon( const odb_SequenceFieldBulkData& bdbCon,
                                 const odb_SequenceInvariant& invars,
                                 odb_String space ,
                                 bool complexData)
{
    cout << space.CStr() << "____ Blocks of Field Output ____" << endl;
    int numBlocks = bdbCon.size();
    cout << space.CStr() << "    number of blocks ";
    cout << numBlocks << endl << endl;
    for (int i=0; i<numBlocks; i++) {
        odbE_printBulkField (bdbCon[i], invars, space+"    ",complexData);    
    }
    cout << endl;
}

void odbE_printBulkField( const odb_FieldBulkData& bulkData,
                          const odb_SequenceInvariant& invars,
                          odb_String space,
                          bool complexData)
{
    const char* blank = " ";
    const odb_Instance& bulkInstance = bulkData.instance();
    odb_String bulkInstanceName = bulkInstance.name();
    cout << space.CStr() << "FieldBulkData for instance: " << 
        bulkInstanceName.CStr() << endl;
    int numFV = bulkData.length();    
    int numComp = bulkData.width();
    float* data = 0;
    double* dataDbl = 0;
    float* conjugateData = 0;
    double* conjugateDataDbl = 0;
    float* lcs = 0;
    double* lcsDbl = 0;
    odb_Enum::odb_PrecisionEnum prec = bulkData.precision();
    if(prec == odb_Enum::SINGLE_PRECISION)
    {
	data = bulkData.data();
	conjugateData = bulkData.conjugateData();
	lcs = bulkData.localCoordSystem();
    }
    else
    {
	dataDbl = bulkData.dataDouble();
	conjugateDataDbl = bulkData.conjugateDataDouble();
	lcsDbl = bulkData.localCoordSystemDouble();
    }
    int nElems = bulkData.numberOfElements();                
    int* elementLabels = bulkData.elementLabels();
    int* nodeLabels = bulkData.nodeLabels();	

    if( nElems && elementLabels )
    {	
	int numIP = numFV/nElems;
	int* integrationPoints = bulkData.integrationPoints();
	const odb_SectionPoint& myBulkSectionPoint = bulkData.sectionPoint();
	int orientationWidth = bulkData.orientationWidth();
        cout << space.CStr() << "Base Element Type : " << bulkData.baseElementType().CStr() << endl;
	if ( myBulkSectionPoint.number() > 0 )
            cout << space.CStr() << "     sectionPoint     : " << myBulkSectionPoint.number() 
                 << ", '" << myBulkSectionPoint.description().CStr() << "'" << endl;
       odb_SequenceString labs = bulkData.componentLabels();
       if(labs.size()){
         
         cout << space.CStr() << "     component Labels: ";
         for (int jj=0;jj<labs.size();jj++)
           cout<<labs[jj].CStr()<<" ";
         cout <<endl;
       }

	int currPos = 0;
	if(prec == odb_Enum::SINGLE_PRECISION)
	{
	    for (int elem = 0;elem<nElems;++elem)
	    {			    	    	    	    
		for (int ip = 0;ip<numIP;ip++,currPos++)
		{
		    cout << space.CStr() << "     element          : "<< 
                        bulkInstanceName.CStr() << "."
			 << elementLabels[currPos] << endl;
                  if(integrationPoints){
                    cout << space.CStr() << "     integrationPoint : " 
                         << integrationPoints[currPos] << endl;
                  }


		    int totPoints = currPos*numComp;
		    cout << space.CStr() <<     "     data             : " ;
		    for (int comp = 0;comp<numComp;++comp)
			cout << data[totPoints++] << blank;                
		    cout << endl;		
		    if (lcs)
		    {
			cout << space.CStr() << "     localCoordSystem : " ;
			int currPtr = currPos*orientationWidth;
			for (int xcs = 0;xcs<orientationWidth;++xcs)
			    cout << lcs[currPtr++] << blank ;
		    }
		    cout << endl;
		}
	    }
	}
	else
	{
	    for (int elem = 0;elem<nElems;++elem)
	    {			    	    	    	    
		for (int ip = 0;ip<numIP;ip++,currPos++)
		{
		    cout << space.CStr() << "     element          : "<< 
                        bulkInstanceName.CStr() << "."
			 << elementLabels[currPos] << endl;
                  if(integrationPoints){
                    cout << space.CStr() << "     integrationPoint : " 
                         << integrationPoints[currPos] << endl;
                  }
		    int totPoints = currPos*numComp;
		    cout << space.CStr() <<     "     data             : " ;
		    for (int comp = 0;comp<numComp;++comp)
			cout << dataDbl[totPoints++] << blank;                
		    cout << endl;		
		    if (lcsDbl)
		    {
			cout << space.CStr() << "     localCoordSystem : " ;
			int currPtr = currPos*orientationWidth;
			for (int xcs = 0;xcs<orientationWidth;++xcs)
			    cout << lcsDbl[currPtr++] << blank ;
		    }
		    cout << endl;
		}
	    }
	}
	if(complexData)
        {
	    cout << space.CStr() << "Conjugate Data " << endl;
            currPos = 0;
	    if(prec == odb_Enum::SINGLE_PRECISION)
	    {
		for (int elem = 0;elem<nElems;++elem)
		{			    		
		    for (int ip = 0;ip<numIP;ip++,++currPos)
		    {
			cout << space.CStr() << "     element          : "<< 
                            bulkInstanceName.CStr() << "."
			     << elementLabels[currPos] << endl;
                     if(integrationPoints){
                       cout << space.CStr() << "     integrationPoint : " 
                            << integrationPoints[currPos] << endl;
                     }
			int totPoints = currPos*numComp;
			cout << space.CStr() << "     conjugate data   : " ;
			for (int comp = 0;comp<numComp;++comp)
			    cout << conjugateData[totPoints++] << blank ;
			cout << endl;
		    }
		}
	    }
	    else
	    {
		for (int elem = 0;elem<nElems;++elem)
		{			    		
		    for (int ip = 0;ip<numIP;ip++,++currPos)
		    {
			cout << space.CStr() << "     element          : "<< 
                            bulkInstanceName.CStr() << "."
			     << elementLabels[currPos] << endl;
                     if(integrationPoints){
                       cout << space.CStr() << "     integrationPoint : " 
                            << integrationPoints[currPos] << endl;
                     }
			int totPoints = currPos*numComp;
			cout << space.CStr() << "     conjugate data   : " ;
			for (int comp = 0;comp<numComp;++comp)
			    cout << conjugateDataDbl[totPoints++] << blank ;
			cout << endl;
		    }
		}
	    }      
	}
	if ( invars.isMember(odb_Enum::MISES) ) 
        {
            currPos = 0;
	    float* bulkMises = bulkData.mises();
	    for (int elem = 0;elem<nElems;++elem)
            {	
                for (int ip = 0;ip<numIP;++ip,++currPos)
                { 		    
                    cout << space.CStr() << "     element          : "<< 
                        bulkInstanceName.CStr() << "."
                         << elementLabels[currPos] << endl;
                    if(integrationPoints){
                      cout << space.CStr() << "     integrationPoint : " 
                           << integrationPoints[currPos] << endl;
                    }
                    cout << space.CStr() <<     "     mises            : " 
                         << bulkMises[currPos] << endl;
                }
            }
        }
    }
    else
    {
	if(prec == odb_Enum::SINGLE_PRECISION)
	{
	    for (int fv = 0;fv<numFV;++fv)
	    {			    
		cout << space.CStr() << "     node             : " << 
                    bulkInstanceName.CStr() //
		     << "." << nodeLabels[fv]  << endl;	   		
		int totPoints = fv*numComp;
		cout << space.CStr() <<     "     data             : " ;
		for (int comp = 0;comp<numComp;++comp)
		    cout << data[totPoints++] << blank;            
		cout << endl;	   
	    }             
	    if (complexData)
	    {
		cout << space.CStr() << " Bulk Conjugate Data " << endl;
		for (int fv = 0;fv<numFV;++fv)
		{			    
		    cout << space.CStr() << "     node             : " << 
                        bulkInstanceName.CStr() //
			 << "." << nodeLabels[fv]  << endl;	   		
		    int totPoints = fv*numComp;
		    cout << space.CStr() <<     "     conjugate data   : " ;
		    for (int comp = 0;comp<numComp;++comp)
			cout << conjugateData[totPoints++] << blank;
		    cout << endl;	   
		}
	    }      
	}
	else
	{
	    for (int fv = 0;fv<numFV;++fv)
	    {			    
		cout << space.CStr() << "     node             : " << 
                    bulkInstanceName.CStr() //
		     << "." << nodeLabels[fv]  << endl;	   		
		int totPoints = fv*numComp;
		cout << space.CStr() <<     "     data             : " ;
		for (int comp = 0;comp<numComp;++comp)
		    cout << dataDbl[totPoints++] << blank;            
		cout << endl;	   
	    }             
	    if (complexData)
	    {
		cout << space.CStr() << " Bulk Conjugate Data " << endl;
		for (int fv = 0;fv<numFV;++fv)
		{			    
		    cout << space.CStr() << "     node             : " << 
                        bulkInstanceName.CStr() //
			 << "." << nodeLabels[fv]  << endl;	   		
		    int totPoints = fv*numComp;
		    cout << space.CStr() <<     "     conjugate data   : " ;
		    for (int comp = 0;comp<numComp;++comp)
			cout << conjugateDataDbl[totPoints++] << blank;
		    cout << endl;	   
		}
	    }
	}
    }
}
void odbE_printFieldValueCon( const odb_SequenceFieldValue& fvCon,
                              const odb_SequenceInvariant& invars,
                              odb_String space,
                              bool complexData)
{
    cout << space.CStr() << "____ FieldValues ____" << endl;
    int numVal = fvCon.size();
    cout << space.CStr() << "    number of fieldValues ";
    cout << numVal << endl << endl;
    for (int i=0; i<numVal; i++) 
        odbE_printFieldValue( fvCon.constGet(i), invars, space+"    ",complexData);    
    cout << endl;
}

void odbE_printFieldLocationCon(odb_SequenceFieldLocation flCon, odb_String space)
{
    cout << space.CStr() << "____ FieldLocations ____ " << endl;
    int numVal = flCon.size();
    for (int i=0; i<numVal; i++)
        odbE_printFieldLocation (flCon.constGet(i), space+"    ");
    cout << endl;
}

void odbE_printFieldValue( const odb_FieldValue& f,
                           const odb_SequenceInvariant& invars,
                           odb_String space,
                           bool complexData)
{
    cout << space.CStr() << "FieldValue : " << endl;
    if( f.elementLabel() >= 0 )
    	cout << space.CStr() << "     element          : "
             << f.instance().name().CStr() << "."
             << f.elementLabel() << endl;
    if( f.nodeLabel() >= 0 )
        cout << space.CStr() << "     node             : " 
             << f.instance().name().CStr() << "."
             << f.nodeLabel()  << endl;
    if( f.sectionPoint().number() > 0 ) {
        cout << space.CStr() << "     sectionPoint     : " 
             << f.sectionPoint().number() << ", '" 
             << f.sectionPoint().description().CStr() << "'" << endl;
    }
    if( f.integrationPoint() > 0 ) 
        cout << space.CStr() << "     integrationPoint : " 
             << f.integrationPoint() << endl;
    
    if ( invars.isMember(odb_Enum::MAGNITUDE) ) 
        cout << space.CStr() <<     "     magnitude        : " << f.magnitude() << endl;    
    if ( invars.isMember(odb_Enum::TRESCA) ) 
        cout << space.CStr() <<     "     tresca           : " << f.tresca() << endl;
    if ( invars.isMember(odb_Enum::PRESS) ) 
        cout << space.CStr() <<     "     press            : " << f.press() << endl;
    if ( invars.isMember(odb_Enum::INV3) ) 
        cout << space.CStr() <<     "     inv3             : " << f.inv3() << endl;
    if ( invars.isMember(odb_Enum::MAX_PRINCIPAL) ) 
        cout << space.CStr() <<     "     maxPrincipal     : " << f.maxPrincipal() << endl;
    if ( invars.isMember(odb_Enum::MID_PRINCIPAL) ) 
        cout << space.CStr() <<     "     midPrincipal     : " << f.midPrincipal() << endl;
    if ( invars.isMember(odb_Enum::MIN_PRINCIPAL) ) 
        cout << space.CStr() <<     "     minPrincipal     : " << f.minPrincipal() << endl;
    if ( invars.isMember(odb_Enum::MAX_INPLANE_PRINCIPAL) ) 
        cout << space.CStr() <<     "     maxInPlanePrincipal     : " 
             << f.maxInPlanePrincipal() << endl;
    if ( invars.isMember(odb_Enum::MIN_INPLANE_PRINCIPAL) ) 
        cout << space.CStr() <<     "     minInPlanePrincipal     : "
             << f.minInPlanePrincipal() << endl;
    if ( invars.isMember(odb_Enum::OUTOFPLANE_PRINCIPAL) ) 
        cout << space.CStr() <<     "     outOfPlanePrincipal     : "
             << f.outOfPlanePrincipal() << endl;
}


void odbE_printFieldLocation( const odb_FieldLocation& f, odb_String space)
{
    cout << space.CStr() << "FieldLocation : " << endl;
    cout << space.CStr() << "     position       : " << f.position() << endl;
    int numSP = f.sectionPoint().size();
    cout << space.CStr() << "     number of sectionPoints  : " << numSP << endl;    
    for (int i=0; i<numSP; i++) 
    {
        const odb_SectionPoint& secP = f.sectionPoint(i);
        cout << space.CStr() << "     " << "    SectionPoint " << secP.number();
        cout << " " << secP.description().CStr() << endl;
    }
    cout << endl;
}


void odbE_printHistoryRegion( odb_HistoryRegion& ho, odb_String space, 
                              int odbE_printLevel)
{
    cout << space.CStr() << "__________HISTORY_REGION '";
    cout << ho.name().CStr() << "' _______________" << endl;
    cout << space.CStr() << "  description  : " << ho.description().CStr() << endl;
    cout << space.CStr() << "  number of output : ";
    cout << ho.historyOutputs().size() << endl;
    
    cout << endl;
    
    if( odbE_printLevel>0 ) 
    {
        --odbE_printLevel;
        odbE_printHistoryPoint (ho.historyPoint(), space+"    ");
        odb_HistoryOutputRepository hvCon = ho.historyOutputs();
        odb_HistoryOutputRepositoryIT hvIter (hvCon);
        for (hvIter.first(); !hvIter.isDone(); hvIter.next()) 
            odbE_printHistoryOutput (hvIter.currentValue(), space+"    ");       
    }
}

void odbE_printHistoryOutput ( const odb_HistoryOutput& h, odb_String space )
{
    cout << space.CStr() << "__________HISTORY_OUTPUT '";
    cout << h.name().CStr() << "' _______________" << endl;
    cout << space.CStr() << "     description : " << h.description().CStr() << endl;
    cout << space.CStr() << "     type        : " << h.type() << endl;
    cout << space.CStr() << "     data        : ";
    odb_SequenceSequenceFloat data = h.data();
    int dSize = data.size();
    for (int i=0; i<dSize; i++) 
    {
        odb_SequenceFloat d = data.constGet(i);
        for (int j=0; j<d.size(); j++) 
            cout << d.constGet(j) << " " ;
        cout << endl;
        if (i<data.size()-1)
            cout << space.CStr() << "                 : " ;
    }
    cout << endl;
}

void odbE_printHistoryPoint( const odb_HistoryPoint& h, odb_String space )
{
    cout << space.CStr() << "    HistoryPoint  " << endl;
    cout << space.CStr() << "            position     : ";
    cout << h.position() << endl;

    if(h.instance().name() != "") {
        cout << space.CStr() << "            instance     : " 
             <<  h.instance().name().CStr() << endl;
    }
    if(h.node().label()>=0) {
        cout << space.CStr() << "            node         : ";
        odbE_printNode(h.node());
    }
    if(h.element().label()>=0) {
        cout << space.CStr() << "            element      : ";
        odbE_printElement(h.element());
    }
    if(h.ipNumber()) {
        cout << space.CStr() << "            ipNumber     : "
             <<  h.ipNumber() << endl;
    }
    if(h.sectionPoint().number()>0) {
        cout << space.CStr() << "            sectionPoint : " 
             << h.sectionPoint().number() << endl;
    }
    if(h.region().name() != "") {
        cout << space.CStr() << "            region       : "
             <<  h.region().name().CStr() << endl;
    }
    cout << endl;
}

void odbE_printSectionCatCon( const odb_SectionCategoryRepository& catCon )
{
    cout << endl; 
    cout << "__________ SECTION_CATEGORY _______________" << endl;
    cout << "    number of categories " << catCon.size() << endl;
    
    odb_SectionCategoryRepositoryIT iter(catCon);
    for (iter.first(); !iter.isDone(); iter.next())
        odbE_printSectionCat (iter.currentValue());
    cout << endl; 
}

void odbE_printSectionCat( const odb_SectionCategory& cat)
{
    cout << endl;
    cout << "    SectionCategory name        : '" << cat.name().CStr()
         << "'" << endl;
    cout << "                    description : '" << cat.description().CStr()
         << "'" << endl;
    int numSP = cat.sectionPoints().size();
    cout << endl << "        " << "sectionpoints : " << numSP << endl;
    for (int i=0; i<numSP; i++) {
        const odb_SectionPoint& secP = cat.sectionPoints(i);
        cout << "        SectionPoint " << secP.number()
             << " " << secP.description().CStr() << endl;
    }
    cout << endl << endl;
}

void odbE_printInteractionCon( const odb_InteractionRepository& IR, odb_Odb& odb)
{
    cout << endl; 
    cout << "__________ INTERACTIONS _______________" << endl;
    cout << "    number of interactions: " << IR.size() << endl;

    odb_String space = "      ";
  odb_InteractionRepositoryIT iter(IR);

  for ( iter.first(); !iter.isDone(); iter.next()) {

    if (odb_isA(odb_SurfaceToSurfaceContactStd,iter.currentValue()))
      {
	cout << space.CStr() << "Surface To Surface Contact Interaction, Standard:\n"<<endl;

	odb_SurfaceToSurfaceContactStd sscS = odb_dynamicCast(odb_SurfaceToSurfaceContactStd,iter.currentValue());
	odbE_printSSC_Std(sscS,odb,space);
      }

    else if (odb_isA(odb_SurfaceToSurfaceContactStd,iter.currentValue()))
      {
	cout << space.CStr() << "Surface To Surface Contact Interaction, Explicit:\n"<<endl;

	odb_SurfaceToSurfaceContactExp sscE = odb_dynamicCast(odb_SurfaceToSurfaceContactExp,iter.currentValue());
	odbE_printSSC_Exp(sscE,odb,space);
      }
    else
      {
	cerr << "Error: Unsupported Interaction Type."<<endl;
      }

  }

}

void odbE_printSSC_Std(const odb_SurfaceToSurfaceContactStd& ssc, 
		       odb_Odb& odb, 
		       odb_String space)
{
  odb_String indent = space;
  indent.Append(space);

  cout << space.CStr() << "sliding: "<< ssc.sliding().CStr() << endl;
  cout << space.CStr() << "smooth: "<< ssc.smooth() << endl;
  cout << space.CStr() << "hcrit: "<< ssc.hcrit() << endl;
  cout << space.CStr() << "limitSlideDistance: "<< ssc.limitSlideDistance() << endl;
  cout << space.CStr() << "slideDistance: "<< ssc.slideDistance() << endl;
  cout << space.CStr() << "extensionZone: "<< ssc.extensionZone() << endl;
  cout << space.CStr() << "adjustMethod: "<< ssc.adjustMethod().CStr() << endl;
  cout << space.CStr() << "adjustTolerance: "<< ssc.adjustTolerance() << endl;
  cout << space.CStr() << "enforcement: "<< ssc.enforcement().CStr() << endl;
  cout << space.CStr() << "thickness: "<< ssc.thickness() << endl;
  cout << space.CStr() << "tied: "<< ssc.tied() << endl;
  cout << space.CStr() << "contactTracking: "<< ssc.contactTracking().CStr() << endl;
  cout << space.CStr() << "createStepName: "<< ssc.createStepName().CStr() << endl;

  odb_String ipName =  ssc.interactionProperty();
  cout << space.CStr() << "interactionProperty: "<< endl;
  odbE_printInteractionProperty(odb.interactionProperties().constGet(ipName),indent);

  cout << space.CStr() << "master surface: "<<endl;
  odb_Set master = ssc.master();
  odbE_printSet(master,indent);

  cout << space.CStr() << "slave surface: "<<endl;
  odb_Set slave = ssc.slave();
  odbE_printSet(slave, indent);

  odb_Set adjust = ssc.adjustSet();
  if(!adjust.name().empty())
    {
      cout << space.CStr() << "adjust set: "<<endl;
      odbE_printSet(adjust, indent);
    }
  
  cout << endl<<endl;
  return;
}

void odbE_printSSC_Exp(const odb_SurfaceToSurfaceContactExp& ssc, 
		  odb_Odb& odb, 
		  odb_String space)
{
  odb_String indent = space;
  indent.Append(space);


  cout << space.CStr() << "sliding:" <<ssc.sliding().CStr()<< endl;
  cout << space.CStr() << "masterNoThick: " << ssc.masterNoThick() << endl;
  cout << space.CStr() << "slaveNoThick: " << ssc.slaveNoThick() << endl;
  cout << space.CStr() << "mechanicalConstraint: " << ssc.mechanicalConstraint().CStr() << endl;
  cout << space.CStr() << "weightingFactorType: " << ssc.weightingFactorType().CStr() << endl;
  cout << space.CStr() << "weightingFactor: " << ssc.weightingFactor() << endl;
  cout << space.CStr() << "createStepName: " << ssc.createStepName().CStr() << endl;

  odb_String ipName =  ssc.interactionProperty();
  cout << space.CStr() << "interactionProperty: "<< endl;
  const odb_IPRepository IPR = odb.interactionProperties();
  odbE_printInteractionProperty(IPR.constGet(ipName),indent);

  cout << space.CStr() << "useReverseDatumAxis: " << ssc.useReverseDatumAxis() << endl;
  cout << space.CStr() << "contactControls: " << ssc.contactControls().CStr() << endl;
  
  odb_Set master = ssc.master();
  cout << space.CStr() << "master surface: "<<endl;
  odbE_printSet(master,indent);

  odb_Set slave = ssc.slave();
  cout << space.CStr() << "slave surface:"<<endl;
  odbE_printSet(slave,indent);
 

  
  return;
}


void odbE_printInteractionProperty (const odb_InteractionProperty& IP,
                                 odb_String space)
{
    if (odb_isA(odb_ContactProperty, IP))
      {
	

	odb_ContactProperty cp = odb_dynamicCast(odb_ContactProperty,IP);
	if (cp.hasValue())
	{
	  odb_TangentialBehavior tang = cp.tangentialBehavior();
          if (tang.hasValue())
          {
              cout << space.CStr() <<  "Contact Property: Tangential Behavior:"<<endl;
              cout << space.CStr() <<  "formulation: " << tang.formulation().CStr() << endl;
              cout << space.CStr() <<  "directionality: " << tang.directionality().CStr() << endl;
              cout << space.CStr() <<  "slipRateDependency: " << tang.slipRateDependency() << endl;
              cout << space.CStr() <<  "pressureDependency: " << tang.pressureDependency() << endl;
              cout << space.CStr() <<  "temperatureDependency: " << tang.temperatureDependency() << endl;
              cout << space.CStr() <<  "dependencies: " << tang.dependencies() << endl;
              cout << space.CStr() <<  "exponentialDecayDefinition: " << tang.exponentialDecayDefinition().CStr() << endl;
              cout << space.CStr() <<  "table: " << endl;
              odb_SequenceSequenceDouble tab = tang.table();
              int r = tab.size();
              for (int row=0;row<r;row++){
                  int c = tab[row].size();
                  cout << space.CStr();
                  for (int col =0;col<c;col++)
                      cout <<  tab[row].constGet(col) << ", " ;
                  cout<<endl;
              }
              cout << space.CStr() <<  endl;
              cout << space.CStr() <<  "shearStressLimit: " << tang.shearStressLimit() << endl;
              cout << space.CStr() <<  "maximumElasticSlip: " << tang.maximumElasticSlip().CStr() << endl;
              cout << space.CStr() <<  "fraction: " << tang.fraction() << endl;
              cout << space.CStr() <<  "absoluteDistance: " << tang.absoluteDistance() << endl;
              cout << space.CStr() <<  "elasticSlipStiffness: " << tang.elasticSlipStiffness() << endl;
              cout << space.CStr() <<  "nStateDependentVars: " << tang.nStateDependentVars() << endl;
              cout << space.CStr() <<  "useProperties: " << tang.useProperties() << endl;
          }
        }
      }
    else
      {
	cout << space.CStr() <<  "Unsupported Interaction Property Type"<<endl;
      }
}

void odbE_printConstraintCon(const odb_ConstraintRepository& CR,
			     odb_Odb& odb)
{

    cout << endl; 
    cout << "__________ CONSTRAINTS _______________" << endl;
    cout << "    number of constraints: " << CR.size() << endl;

    odb_String space = "      ";

    odb_ConstraintRepositoryIT iter(CR);
    int i = 1;

  for ( iter.first(); !iter.isDone(); iter.next()) {
    cout << "____ Constraint #" << i++<< " ____" << endl;
    if (odb_isA(odb_Tie,iter.currentValue()))
      {
	cout << space.CStr() << "Tie Constraint:"<<endl;

	odb_Tie tie = odb_dynamicCast(odb_Tie,iter.currentValue());
	odbE_printTie(tie,odb,space);
      }
    else if (odb_isA(odb_DisplayBody,iter.currentValue()))
      {
	cout << space.CStr() << "Display Body Constraint:"<<endl;

	odb_DisplayBody db = odb_dynamicCast(odb_DisplayBody,iter.currentValue());
	odbE_printDisplayBody(db,odb,space);
      }
    else if (odb_isA(odb_Coupling,iter.currentValue()))
      {
          cout  << space.CStr() << "Coupling Constraint:"<<endl;
          
          odb_Coupling coup = odb_dynamicCast(odb_Coupling,iter.currentValue());
          odbE_printCoupling(coup, odb, space);
      }
    else if (odb_isA(odb_MPC,iter.currentValue()))
      {
          cout  << space.CStr() << "MPC Constraint:"<<endl;
          
          odb_MPC mpc = odb_dynamicCast(odb_MPC,iter.currentValue());
          odbE_printMPC(mpc, odb, space);
      }
    else if (odb_isA(odb_ShellSolidCoupling,iter.currentValue()))
      {
          cout  << space.CStr() << "Shell Solid Coupling Constraint:"<<endl;
          
          odb_ShellSolidCoupling coup = odb_dynamicCast(odb_ShellSolidCoupling,iter.currentValue());
          odbE_printShellSolidCoupling(coup, odb, space);
      }    
    else
      {
	cout << "Unsupported Constraint type for constraint: "<<iter.currentKey().CStr()<<endl;
      }

    cout << endl<<endl;
  }
  cout << "____________________________________________"<<endl;
}

void odbE_printTie(const odb_Tie& tie,
		   odb_Odb& odb,
		   odb_String space)
{
	if (tie.hasValue())
	{
	    odb_Set master = tie.master();
	    cout << space.CStr() << "Master Surf: "<<master.name().CStr()<<endl;
	    odb_String indent = space;
	    indent.Append(space);

	    odbE_printSet(master,indent);

	    odb_Set slave =tie.slave();
	    cout << space.CStr() << "Slave Surf: "<<slave.name().CStr()<<endl;
	    odbE_printSet(slave,indent);
	    if (tie.adjust()) 
	      {
		cout << space.CStr() << "Slave position will be ADJUSTED"<<endl;
		
		odb_String tolMeth = tie.positionToleranceMethod();

		cout << space.CStr() << "Position Tolerance will be "<< tolMeth.CStr();
		if (tolMeth == "SPECIFIED")
		  cout << space.CStr() << " as: "<<tie.positionTolerance();
		cout << space.CStr() <<endl;
	      }
	    if(tie.tieRotations())
	      cout << space.CStr() << "Rotations are tied."<<endl;

   	    odb_String crMeth =  tie.constraintRatioMethod();
	    cout << space.CStr() << "Constraint Ratio is "<<crMeth.CStr()<<endl;
	    cout << space.CStr() << "      "<< tie.constraintRatio() <<endl;
	    cout << space.CStr() << "Constraint Enforcement is: "<< tie.constraintEnforcement().CStr()<<endl;
	    cout << space.CStr() << "Thickness: "<< tie.thickness() <<endl;

	}
}


void odbE_printDisplayBody(const odb_DisplayBody& displayBody,
		   odb_Odb& odb,
		   odb_String space)
{
	if (displayBody.hasValue())
	{
	    odb_String instanceName = displayBody.instanceName();
            cout << space.CStr() << "Instance Name: " << instanceName.CStr() << endl;
            cout << space.CStr() << "Reference Node 1: " << displayBody.referenceNode1InstanceName().CStr();
            cout << "." << displayBody.referenceNode1Label()<< endl;
            cout << space.CStr() << "Reference Node 2: " << displayBody.referenceNode2InstanceName().CStr();
            cout << "." << displayBody.referenceNode2Label()<< endl;
            cout << space.CStr() << "Reference Node 3: " << displayBody.referenceNode3InstanceName().CStr();                     cout << "." << displayBody.referenceNode3Label()<< endl;

	}
}

void odbE_printCoupling(const odb_Coupling& coup, 
                        odb_Odb& odb,
                        odb_String space)
{
	if (coup.hasValue())
	{
	    odb_Set surface  = coup.surface();
	    cout  <<  space.cStr() << "Surface: "<<surface.name().cStr()<<endl;
            odb_String indent = space;
	    indent.Append(space);
	    odbE_printSet(surface ,indent);	    

	    odb_Set refPoint =coup.refPoint();
	    cout  <<  space.cStr() << "RefPoint: "<<refPoint.name().cStr()<<endl;
	    odbE_printSet(refPoint, indent);

           
	    cout  << space.cStr() << "Coupling Type: " << coup.couplingType().cStr()<<endl;
	    cout  << space.cStr() << "Weighting Method: " << coup.weightingMethod().cStr()<<endl;
	    cout  << space.cStr() << "Influence Radius: "<< coup.influenceRadius() <<endl;

            if(coup.u1()) cout << "U1 constrained"<<endl;
            if(coup.u2()) cout << "U2 constrained"<<endl;
            if(coup.u3()) cout << "U3 constrained"<<endl;
            if(coup.ur1()) cout << "UR1 constrained"<<endl;
            if(coup.ur2()) cout << "UR2 constrained"<<endl;
            if(coup.ur3()) cout << "UR3 constrained"<<endl;
            
	    odb_Set nodes  = coup.couplingNodes();
	    cout  << space.cStr() << "Coupling Node Set: "<<nodes.name().cStr()<< endl;
	    odbE_printSet(nodes);

	}
}

void odbE_printMPC(const odb_MPC& mpc, 
                   odb_Odb& odb,
                   odb_String space)
{
	if (mpc.hasValue())
	{
	    odb_Set surface  = mpc.surface();
	    cout  <<  space.cStr() << "Surface: "<<surface.name().cStr()<<endl;
            odb_String indent = space;
	    indent.Append(space);
	    odbE_printSet(surface ,indent);	    

	    odb_Set refPoint =mpc.refPoint();
	    cout  <<  space.cStr() << "RefPoint: "<<refPoint.name().cStr()<<endl;
	    odbE_printSet(refPoint, indent);
           
	    cout  << space.cStr() << "MPC Type: " << mpc.mpcType().cStr()<<endl;
	    cout  << space.cStr() << "User Mode: " << mpc.userMode().cStr()<<endl;
	    cout  << space.cStr() << "User Type: "<< mpc.userType() <<endl;
	}
}
void odbE_printShellSolidCoupling(const odb_ShellSolidCoupling& coup, 
                                  odb_Odb& odb,
                                  odb_String space)
{
	if (coup.hasValue())
	{
	    odb_Set shellEdge  = coup.shellEdge();
            cout  <<  space.cStr() << "Shell Edge: "<<shellEdge.name().cStr()<<endl;
	    odb_String indent = space;
	    indent.Append(space);
	    odbE_printSet(shellEdge ,indent);	    

	    odb_Set solidFace = coup.solidFace();
	    cout  <<  space.cStr() << "Solid Face: "<<solidFace.name().cStr()<<endl;
	    odbE_printSet(solidFace, indent);

           
	    cout  << space.cStr() << "Position Tolerance Method: " << coup.positionToleranceMethod().cStr()<<endl;
	    cout  << space.cStr() << "Position Tolerance: " << coup.positionTolerance()<<endl;
	    cout  << space.cStr() << "Influence Distance Method: "<< coup.influenceDistanceMethod().cStr() <<endl;
            cout  << space.cStr() << "Influence Distance: "<< coup.influenceDistance() <<endl;           

	}
}

// Begin local includes
#include <odb_MaterialTypes.h>
#include <odb_SectionTypes.h>
#include <odb_WipReporter.h>
#include <nex_Exception.h>
// End local includes

int ABQmain(int argc, char** argv);
int main(int argc, char** argv)
{ 
    odb_initializeAPI(); 

    int status = 0; 
    try { 
        status = ABQmain(argc, argv);
    } 
    catch (const nex_Exception& nex) {
    status = 1;
    fprintf(stderr, "%s\n", nex.UserReport().CStr()); 
    fprintf(stderr, "ODB Application exited with error(s)\n");
    } 
    catch (...) {
    status = 1;
    fprintf(stderr, "ODB Application exited with error(s)\n");
    } 
    odb_finalizeAPI(); 
    return (status);
} 

