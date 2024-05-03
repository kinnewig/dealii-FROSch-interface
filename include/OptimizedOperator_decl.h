#ifndef _FROSCH_OPTIMIZEDOPERATOR_DECL_HPP
#define _FROSCH_OPTIMIZEDOPERATOR_DECL_HPP

#include <FROSch_OverlappingOperator_def.hpp>

namespace FROSch {

    using namespace std;
    using namespace Teuchos;
    using namespace Xpetra;



    template <class SC = double,
              class LO = int,
              class GO = DefaultGlobalOrdinal,
              class NO = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType>
    class OptimizedSchwarzOperator : public OverlappingOperator<SC,LO,GO,NO> {

    protected:
        using CommPtr                 = typename SchwarzOperator<SC,LO,GO,NO>::CommPtr;

        using XMultiVector            = typename SchwarzOperator<SC,LO,GO,NO>::XMultiVector;
        using XMultiVectorPtr         = typename SchwarzOperator<SC,LO,GO,NO>::XMultiVectorPtr;

        template <typename S>
        using XMultiVectorTemplate    = Xpetra::MultiVector<S,LO,GO,NO>;
        template <typename S>
        using XMultiVectorTemplatePtr = RCP<XMultiVectorTemplate<S>>;

        using GraphPtr                = typename SchwarzOperator<SC,LO,GO,NO>::GraphPtr;

        using XMapPtr                 = typename SchwarzOperator<SC,LO,GO,NO>::XMapPtr;
        using ConstXMapPtr            = typename SchwarzOperator<SC,LO,GO,NO>::ConstXMapPtr;

        using XMatrix                 = typename SchwarzOperator<SC,LO,GO,NO>::XMatrix;
        using XMatrixPtr              = typename SchwarzOperator<SC,LO,GO,NO>::XMatrixPtr;
        using ConstXMatrixPtr         = typename SchwarzOperator<SC,LO,GO,NO>::ConstXMatrixPtr;

        using ConstXCrsGraphPtr       = typename SchwarzOperator<SC,LO,GO,NO>::ConstXCrsGraphPtr;

        using ParameterListPtr        = typename SchwarzOperator<SC,LO,GO,NO>::ParameterListPtr;

    public:

        /**
         * Default constructor
         */
        OptimizedSchwarzOperator(ConstXMatrixPtr  k,
                                 ParameterListPtr parameterList);

        /**
         * Not implemented. 
         */
        virtual int 
        initialize() override;

        int 
        initialize(int      overlap,
                   GraphPtr dualGraph);

        int 
        continue_initialize(XMapPtr overlappingMap);

        int 
        compute() override;

        int 
        compute(ConstXMatrixPtr neumannMatrix, ConstXMatrixPtr robinMatrix);

        /**
         * TODO: Temporary work arround, remove later!
         */
       	virtual void apply(const XMultiVector &x,
                           XMultiVector &y,
                           ETransp mode=NO_TRANS,
                           SC alpha=ScalarTraits<SC>::one(),
                           SC beta=ScalarTraits<SC>::zero()) const override
        {
        	OverlappingOperator<SC,LO,GO,NO>::apply(x,y,true,mode,alpha,beta);
        };

        int 
        communicateOverlappingTriangulation(
            XMultiVectorPtr                     nodeList,
            XMultiVectorTemplatePtr<long long>  elementList,
            XMultiVectorTemplatePtr<long long>  auxillaryList,
            XMultiVectorPtr                    &nodeListOverlapping,
            XMultiVectorTemplatePtr<long long> &elementListOverlapping,
            XMultiVectorTemplatePtr<long long> &auxillaryListOverlapping);

        void 
        describe(FancyOStream          &out,
                 const EVerbosityLevel  verbLevel) const override;

        string 
        description() const override;

    protected:
        
        int
        buildOverlappingMap(
            int      overlap,
            GraphPtr dualGraph);

        int 
        updateLocalOverlappingMatrices() override;

        int 
        updateLocalOverlappingMatrices_Symbolic();

        void 
        extractLocalSubdomainMatrix_Symbolic();

        /**
         *  Store a RCP<Xpetra::CrsGraph<SC,LO,GO,NO>>
         *  which contains the dual graph, i.e.
         *  if there is an entry in (row i, column j)
         *  element i and element j are neighbors.
         */
        GraphPtr DualGraph_;

        /*
         * The DualGraph with overlap
         */
        ConstXCrsGraphPtr OverlappingGraph_;

        /**
         * The column map of the DualGraph with overlap
         */
        ConstXMapPtr OverlappingElementMap_;

        ConstXMatrixPtr NeumannMatrix_;
        ConstXMatrixPtr RobinMatrix_;

    };

} //namespace FROSch

#endif
