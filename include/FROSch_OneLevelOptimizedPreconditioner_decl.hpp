//@HEADER
// ************************************************************************
//
//               ShyLU: Hybrid preconditioner package
//                 Copyright 2012 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Alexander Heinlein (alexander.heinlein@uni-koeln.de)
//
// ************************************************************************
//@HEADER

#ifndef _FROSCH_ONELEVELOPTIMIZEDPRECONDITIONER_DECL_HPP
#define _FROSCH_ONELEVELOPTIMIZEDPRECONDITIONER_DECL_HPP

#include <FROSch_SchwarzPreconditioner_decl.hpp>
#include <FROSch_SchwarzPreconditioner_def.hpp>
#include "FROSch_OptimizedOperator_decl.h"
#include "FROSch_OptimizedOperator_def.h"


namespace FROSch {

    using namespace std;
    using namespace Teuchos;
    using namespace Xpetra;    

    template <class SC = double,
              class LO = int,
              class GO = DefaultGlobalOrdinal,
              class NO = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType>
    class OneLevelOptimizedPreconditioner : public SchwarzPreconditioner<SC,LO,GO,NO> {

    protected:

        using XMapPtr                           = typename SchwarzPreconditioner<SC,LO,GO,NO>::XMapPtr;
        using ConstXMapPtr                      = typename SchwarzPreconditioner<SC,LO,GO,NO>::ConstXMapPtr;

        using GraphPtr                          = RCP<Xpetra::CrsGraph<LO,GO,NO>>;

        using XMatrixPtr                        = typename SchwarzPreconditioner<SC,LO,GO,NO>::XMatrixPtr;
        using ConstXMatrixPtr                   = typename SchwarzPreconditioner<SC,LO,GO,NO>::ConstXMatrixPtr;

        using XMultiVector                      = typename SchwarzPreconditioner<SC,LO,GO,NO>::XMultiVector;
        using XMultiVectorPtr                   = typename SchwarzPreconditioner<SC,LO,GO,NO>::XMultiVectorPtr;

        template <typename S>
        using XMultiVectorTemplate              = Xpetra::MultiVector<S,LO,GO,NO>;
        template <typename S>
        using XMultiVectorTemplatePtr           = RCP<XMultiVectorTemplate<S>>;

        using ParameterListPtr                  = typename SchwarzPreconditioner<SC,LO,GO,NO>::ParameterListPtr;

        using SumOperatorPtr                    = typename SchwarzPreconditioner<SC,LO,GO,NO>::SumOperatorPtr;
        using MultiplicativeOperatorPtr         = typename SchwarzPreconditioner<SC,LO,GO,NO>::MultiplicativeOperatorPtr;

        using OverlappingOperatorPtr            = typename SchwarzPreconditioner<SC,LO,GO,NO>::OverlappingOperatorPtr;
        using OptimizedOperatorPtr              = RCP<OptimizedSchwarzOperator<SC,LO,GO,NO>>;

    public:

        OneLevelOptimizedPreconditioner(ConstXMatrixPtr  k,
                                        GraphPtr         dualGraph,
                                        ParameterListPtr parameterList);

        /**
         * Not implemented. 
         */
        virtual int 
        initialize(bool useDefault) override;

        int 
        initialize(XMapPtr overlappingMap);

        /**
         * Not implemented. 
         */
        virtual int 
        compute() override;

        int 
        compute(ConstXMatrixPtr neumannMatrix, 
                ConstXMatrixPtr robinMatrix);

        virtual void 
        apply(const XMultiVector &x,
              XMultiVector &y,
              ETransp       mode  = NO_TRANS,
              SC            alpha = ScalarTraits<SC>::one(),
              SC            beta  = ScalarTraits<SC>::zero()) const override;

        virtual const ConstXMapPtr 
        getDomainMap() const override;

        virtual const ConstXMapPtr 
        getRangeMap() const override;

        int 
        communicateOverlappingTriangulation(
            XMultiVectorPtr                     nodeList,
            XMultiVectorTemplatePtr<long long>  elementList,
            XMultiVectorTemplatePtr<long long>  auxillaryList,
            XMultiVectorPtr                    &nodeListOverlapping,
            XMultiVectorTemplatePtr<long long> &elementListOverlapping,
            XMultiVectorTemplatePtr<long long> &auxillaryListOverlapping);

        virtual void 
        describe(
            FancyOStream          &out,
            const EVerbosityLevel  verbLevel = Describable::verbLevel_default) const override;

        virtual string 
        description() const override;

        virtual int 
        resetMatrix(ConstXMatrixPtr &k);

    protected:

        ConstXMatrixPtr           K_;

        SumOperatorPtr            SumOperator_;
        MultiplicativeOperatorPtr MultiplicativeOperator_;
        OverlappingOperatorPtr    OverlappingOperator_;
        bool                      UseMultiplicative_ = false;
    };

}

#endif
