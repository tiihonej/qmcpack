//////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Miguel A. Morales, moralessilva2@llnl.gov
//    Lawrence Livermore National Laboratory
//
// File created by:
// Miguel A. Morales, moralessilva2@llnl.gov
//    Lawrence Livermore National Laboratory
////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_AFQMC_HAMILTONIANOPERATIONS_REAL3INDEXFACTORIZATION_BATCHED_V2_HPP
#define QMCPLUSPLUS_AFQMC_HAMILTONIANOPERATIONS_REAL3INDEXFACTORIZATION_BATCHED_V2_HPP

#include <vector>
#include <type_traits>
#include <random>

#include "Configuration.h"
#include "AFQMC/config.h"
#include "multi/array.hpp"
#include "multi/array_ref.hpp"
#include "AFQMC/Numerics/ma_operations.hpp"

#include "AFQMC/Utilities/type_conversion.hpp"
#include "AFQMC/Utilities/taskgroup.h"
#include "AFQMC/Utilities/Utils.hpp"
#include "AFQMC/Numerics/helpers/batched_operations.hpp"
#include "AFQMC/Numerics/tensor_operations.hpp"

namespace qmcplusplus
{

namespace afqmc
{

// Custom implementation for real build
class Real3IndexFactorization_batched_v2
{

  // allocators
  using Allocator = device_allocator<ComplexType>;
  using SpAllocator = device_allocator<SPComplexType>;
  using SpRAllocator = device_allocator<SPRealType>;
  using Allocator_shared = node_allocator<ComplexType>;
  using SpAllocator_shared = node_allocator<SPComplexType>;
  using SpRAllocator_shared = node_allocator<SPRealType>;

  // type defs
  using pointer = typename Allocator::pointer;
  using sp_pointer = typename SpAllocator::pointer;
  using sp_rpointer = typename SpRAllocator::pointer;
  using pointer_shared = typename Allocator_shared::pointer;
  using sp_pointer_shared = typename SpAllocator_shared::pointer;
  using sp_rpointer_shared = typename SpRAllocator_shared::pointer;

  using CVector = ComplexVector<Allocator>; 
  using SpVector = SPComplexVector<SpAllocator>; 
  using SpCMatrix = SPComplexMatrix<SpAllocator>; 
  using CVector_ref = ComplexVector_ref<pointer>; 
  using SpCVector_ref = SPComplexVector_ref<sp_pointer>; 
  using CMatrix_ref = ComplexMatrix_ref<pointer>; 
  using SpCMatrix_ref = SPComplexMatrix_ref<sp_pointer>; 
  using SpRMatrix_ref = SPComplexMatrix_ref<sp_rpointer>; 
  using SpCTensor_ref = boost::multi::array_ref<SPComplexType,3,sp_pointer>;
  using SpC4Tensor_ref = boost::multi::array_ref<SPComplexType,4,sp_pointer>;
  using C4Tensor_ref = boost::multi::array_ref<ComplexType,4,pointer>;

  using shmCMatrix = ComplexMatrix<Allocator_shared>;
  using shmSpC3Tensor = SPComplex3Tensor<SpAllocator_shared>;
  using shmSpCMatrix = SPComplexMatrix<SpAllocator_shared>; 
  using shmSpRMatrix = boost::multi::array<SPRealType,2,SpRAllocator_shared>; 

  using mpi3RMatrix = boost::multi::array<RealType,2,shared_allocator<RealType>>;
  using mpi3CMatrix = boost::multi::array<ComplexType,2,shared_allocator<ComplexType>>;

  public:

    static const HamiltonianTypes HamOpType = RealDenseFactorized;
    HamiltonianTypes getHamType() const { return HamOpType; }

    template<class shmCMatrix_, class shmSpRMatrix_, class shmSpC3Tensor_>
    Real3IndexFactorization_batched_v2(WALKER_TYPES type,
                 mpi3RMatrix&& hij_,
                 shmCMatrix_&& haj_,
                 shmSpRMatrix_&& vik,
                 std::vector<shmSpC3Tensor_>&& vnak,
                 mpi3CMatrix&& vn0_,
                 ValueType e0_,
                 Allocator const& alloc_,
                 int cv0,
                 int gncv,
                 long maxMem = 2000):
        allocator_(alloc_),
        sp_allocator_(alloc_),
        walker_type(type),
        max_memory_MB(maxMem),
        global_origin(cv0),
        global_nCV(gncv),
        local_nCV(0),
        E0(e0_),
        hij(std::move(hij_)),
        hij_dev(hij.extensions(),allocator_),
        haj(std::move(haj_)),
        Likn(std::move(vik)),
        Lnak(std::move(move_vector<shmSpC3Tensor>(std::move(vnak)))),
        vn0(std::move(vn0_)),
        TBuff(iextensions<1u>{1},sp_allocator_)
    {
      local_nCV=Likn.size(1);
      size_t lnak(0);
      for(auto& v: Lnak) lnak += v.num_elements();
      for(int i=0; i < hij.size(0); i++) {
        for(int j=0; j < hij.size(1); j++) {
          hij_dev[i][j] = ComplexType(hij[i][j]);
        }
      }
      app_log()<<"****************************************************************** \n"
               <<"  Static memory usage by Real3IndexFactorization_batched_v2 (node 0 in MB) \n"
               <<"  Likn: " <<Likn.num_elements()*sizeof(SPRealType)/1024.0/1024.0 <<" \n"
               <<"  Lnak: " <<lnak*sizeof(SPComplexType)/1024.0/1024.0 <<" \n"
               <<"  Buffer memory limited to (not yet allocated) :" <<max_memory_MB <<" MB. \n";
      memory_report();
    }

    ~Real3IndexFactorization_batched_v2() {}

    Real3IndexFactorization_batched_v2(const Real3IndexFactorization_batched_v2& other) = delete;
    Real3IndexFactorization_batched_v2& operator=(const Real3IndexFactorization_batched_v2& other) = delete;
    Real3IndexFactorization_batched_v2(Real3IndexFactorization_batched_v2&& other) = default;
    Real3IndexFactorization_batched_v2& operator=(Real3IndexFactorization_batched_v2&& other) = default;

    boost::multi::array<ComplexType,2> getOneBodyPropagatorMatrix(TaskGroup_& TG, boost::multi::array<ComplexType,1> const& vMF) 
    {
      int NMO = hij.size(0);
      // in non-collinear case with SO, keep SO matrix here and add it
      // for now, stay collinear

      CVector vMF_(vMF);
      CVector P1D(iextensions<1u>{NMO*NMO});
      fill_n(P1D.origin(),P1D.num_elements(),ComplexType(0));
      vHS(vMF_, P1D);
      if(TG.TG().size() > 1)
        TG.TG().all_reduce_in_place_n(to_address(P1D.origin()),P1D.num_elements(),std::plus<>());

      boost::multi::array<ComplexType,2> P1({NMO,NMO});
      copy_n(P1D.origin(),NMO*NMO,P1.origin());

      using ma::conj;

      for(int i=0; i<NMO; i++) {
        P1[i][i] += hij[i][i] + vn0[i][i];
        for(int j=i+1; j<NMO; j++) {
          P1[i][j] += hij[i][j] + vn0[i][j];
          P1[j][i] += hij[j][i] + vn0[j][i];
          // This is really cutoff dependent!!!
          if( std::abs( P1[i][j] - ma::conj(P1[j][i]) ) > 1e-6 ) {
            app_error()<<" WARNING in getOneBodyPropagatorMatrix. P1 is not hermitian. \n";
            app_error()<<i <<" " <<j <<" " <<P1[i][j] <<" " <<P1[j][i] <<" "
                       <<hij[i][j] <<" " <<hij[j][i] <<" "
                       <<vn0[i][j] <<" " <<vn0[j][i] <<std::endl;
            //APP_ABORT("Error in getOneBodyPropagatorMatrix. P1 is not hermitian. \n");
          }
          P1[i][j] = 0.5*(P1[i][j]+ma::conj(P1[j][i]));
          P1[j][i] = ma::conj(P1[i][j]);
        }
      }

      return P1;
    }

    template<class Mat, class MatB>
    void energy(Mat&& E, MatB const& G, int k, bool addH1=true, bool addEJ=true, bool addEXX=true) {
      MatB* Kr(nullptr);
      MatB* Kl(nullptr);
      energy(E,G,k,Kl,Kr,addH1,addEJ,addEXX);
    }

    // KEleft and KEright must be in shared memory for this to work correctly
    template<class Mat, class MatB, class MatC, class MatD>
    void energy(Mat&& E, MatB const& Gc, int nd, MatC* KEleft, MatD* KEright, bool addH1=true, bool addEJ=true, bool addEXX=true) {
      assert(E.size(1)>=3);
      assert(nd >= 0);
      assert(nd < haj.size());
      if(walker_type==COLLINEAR)
        assert(2*nd+1 < Lnak.size());
      else
        assert(nd < Lnak.size());

      int nwalk = Gc.size(0);
      int nspin = (walker_type==COLLINEAR?2:1);
      int NMO = hij.size(0); 
      int nel[2];
      nel[0] = Lnak[nspin*nd].size(1);
      nel[1] = ((nspin==2)?Lnak[nspin*nd+1].size(1):0);
      assert(Lnak[nspin*nd].size(0) == local_nCV);
      assert(Lnak[nspin*nd].size(2) == NMO);
      if(nspin==2) {
        assert(Lnak[nspin*nd+1].size(0) == local_nCV);
        assert(Lnak[nspin*nd+1].size(2) == NMO);
      }
      assert(Gc.num_elements() == nwalk*(nel[0]+nel[1])*NMO);

      int getKr = KEright!=nullptr;
      int getKl = KEleft!=nullptr;
      if(E.size(0) != nwalk || E.size(1) < 3)
        APP_ABORT(" Error in AFQMC/HamiltonianOperations/Real3IndexFactorization_batched_v2::energy(...). Incorrect matrix dimensions \n");

      // T[nwalk][nup][nup][local_nCV] + D[nwalk][nwalk][local_nCV]
      size_t mem_needs(0);
      size_t cnt(0);
      if(addEJ) {
#if MIXED_PRECISION
        mem_needs += nwalk*local_nCV; 
#else
        if(not getKl) mem_needs += nwalk*local_nCV;
#endif
      }
      if(addEXX) {
#if MIXED_PRECISION
        mem_needs += nwalk*nel[0]*NMO;
#else
        if(nspin == 2) mem_needs += nwalk*nel[0]*NMO;
#endif
      }
      int max_nCV = 0;
      if(addEXX){
        long LBytes = max_memory_MB*1024L*1024L - mem_needs*sizeof(SPComplexType);
        int Bytes = int(LBytes / long(nwalk*nel[0]*nel[0]*sizeof(SPComplexType)));
        max_nCV = std::min( std::max(1, Bytes), local_nCV);
        assert(max_nCV > 1 && max_nCV <= local_nCV);
        mem_needs += long(max_nCV*nwalk*nel[0]*nel[0]);
      }
      set_buffer(mem_needs);

      // messy
      sp_pointer Klptr(nullptr);
      long Knr=0, Knc=0;
      if(addEJ) {
        Knr=nwalk;
        Knc=local_nCV;
        if(getKr) {
          assert(KEright->size(0) == nwalk && KEright->size(1) == local_nCV);
          assert(KEright->stride(0) == KEright->size(1));
        }
#if MIXED_PRECISION
        if(getKl) {
          assert(KEleft->size(0) == nwalk && KEleft->size(1) == local_nCV);
          assert(KEleft->stride(0) == KEleft->size(1));
        }
#else
        if(getKl) {
          assert(KEleft->size(0) == nwalk && KEleft->size(1) == local_nCV);
          assert(KEleft->stride(0) == KEleft->size(1));
          Klptr = make_device_ptr(KEleft->origin());
        } else 
#endif
        {
          Klptr = TBuff.origin()+cnt;
          cnt += Knr*Knc; 
        }
        fill_n(Klptr,Knr*Knc,SPComplexType(0.0));
      } else if(getKr or getKl) {
        APP_ABORT(" Error: Kr and/or Kl can only be calculated with addEJ=true.\n");
      }
      SpCMatrix_ref Kl(Klptr,{long(Knr),long(Knc)});

      for(int n=0; n<nwalk; n++)
        std::fill_n(E[n].origin(),3,ComplexType(0.));


      // one-body contribution
      // haj[ndet][nocc*nmo]
      // not parallelized for now, since it would require customization of Wfn
      if(addH1) {
        CVector_ref haj_ref(make_device_ptr(haj[nd].origin()), iextensions<1u>{haj[nd].num_elements()});
        ma::product(ComplexType(1.),Gc,haj_ref,ComplexType(1.),E(E.extension(0),0));
        for(int i=0; i<nwalk; i++)
          E[i][0] += E0;
      }

      // move calculation of H1 here
      // NOTE: For CLOSED/NONCOLLINEAR, can do all walkers simultaneously to improve perf. of GEMM
      //       Not sure how to do it for COLLINEAR.
      if(addEXX) {
        SPRealType scl = (walker_type==CLOSED?2.0:1.0);

        for(int ispin=0, is0=0; ispin<nspin; ispin++) {

          size_t cnt_(cnt);
          sp_pointer ptr(nullptr);
#if MIXED_PRECISION
          ptr = TBuff.origin()+cnt_;
          cnt_ += nwalk*nel[ispin]*NMO;
          for(int n=0; n<nwalk; ++n) {
            copy_n_cast(make_device_ptr(Gc[n].origin())+is0,nel[ispin]*NMO,ptr+n*nel[ispin]*NMO);  
          }
#else
          if(nspin==1) {
            ptr = make_device_ptr(Gc.origin());
          } else {
            ptr = TBuff.origin()+cnt_;
            cnt_ += nwalk*nel[ispin]*NMO;
            for(int n=0; n<nwalk; ++n) {
              using std::copy_n;
              copy_n(make_device_ptr(Gc[n].origin())+is0,nel[ispin]*NMO,ptr+n*nel[ispin]*NMO);  
            }
          }
#endif
 
          SpCMatrix_ref GF(ptr,{nwalk*nel[ispin],NMO});  

          int nCV=0;
          while( nCV < local_nCV) {

            int nvecs = std::min(local_nCV-nCV,max_nCV);
            SpCMatrix_ref Lna(make_device_ptr(Lnak[nd*nspin + ispin][nCV].origin()),
                                                   {nvecs*nel[ispin],NMO});
            SpCMatrix_ref Twbna(TBuff.origin()+cnt_,{nwalk*nel[ispin],nvecs*nel[ispin]});
            SpC4Tensor_ref T4Dwbna(Twbna.origin(),{nwalk,nel[ispin],nvecs,nel[ispin]});
  
            ma::product(GF,ma::T(Lna),Twbna);

            using ma::dot_wanb;
            dot_wanb(nwalk,nel[ispin],nvecs,SPComplexType(-0.5*scl),Twbna.origin(),to_address(E[0].origin())+1,E.stride(0));

            if(addEJ) {
              using ma::Tanb_to_Kl;
              Tanb_to_Kl(nwalk,nel[ispin],nvecs,local_nCV,Twbna.origin(),Kl.origin()+nCV);
            }

            nCV += max_nCV;

          }
          is0 += nel[ispin]*NMO; 

        } // ispin 
      }   

      if(addEJ) {
        if(not addEXX) {
          // calculate Kr
          APP_ABORT(" Error: Finish addEJ and not addEXX");
        }
        SPRealType scl = (walker_type==CLOSED?2.0:1.0);
        for(int n=0; n<nwalk; ++n) 
          E[n][2] += 0.5*static_cast<ComplexType>(scl*scl*ma::dot(Kl[n],Kl[n]));
#if MIXED_PRECISION
        if(getKl) 
          copy_n_cast(Klptr,KEleft->num_elements(),make_device_ptr(KEleft->origin()));
#endif
        if(getKr) 
          copy_n_cast(Klptr,KEright->num_elements(),make_device_ptr(KEright->origin()));
      }
    }

    template<class... Args>
    void fast_energy(Args&&... args)
    {
      APP_ABORT(" Error: fast_energy not implemented in Real3IndexFactorization_batched_v2. \n");
    }

    template<class MatA, class MatB,
             typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality==1)>,
             typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality==1)>,
             typename = void
            >
    void vHS(MatA& X, MatB&& v, double a=1., double c=0.) {
      assert( Likn.size(1) == X.size(0) );
      assert( Likn.size(0) == v.size(0) );
#if MIXED_PRECISION
      size_t mem_needs = X.num_elements()+v.num_elements();
      set_buffer(mem_needs);
      SpCVector_ref vsp(TBuff.origin(), v.extensions()); 
      SpCVector_ref Xsp(vsp.origin()+vsp.num_elements(), X.extensions());
      copy_n_cast(make_device_ptr(X.origin()),X.num_elements(),Xsp.origin());
      if( std::abs(c-0.0) > 1e-6 )
        copy_n_cast(make_device_ptr(v.origin()),v.num_elements(),vsp.origin());
      ma::product(SPValueType(a),Likn,Xsp,SPValueType(c),vsp);
      copy_n_cast(vsp.origin(),vsp.num_elements(),make_device_ptr(v.origin()));
#else 
      ma::product(SPValueType(a),Likn,X,SPValueType(c),v);
#endif
    }

    template<class MatA, class MatB,
             typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality==2)>,
             typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality==2)>
            >
    void vHS(MatA& X, MatB&& v, double a=1., double c=0.) {
      assert( Likn.size(1) == X.size(0) );
      assert( Likn.size(0) == v.size(0) );
      assert( X.size(1) == v.size(1) );
#if MIXED_PRECISION
      size_t mem_needs = X.num_elements()+v.num_elements();
      set_buffer(mem_needs);
      SpCMatrix_ref vsp(TBuff.origin(), v.extensions());
      SpCMatrix_ref Xsp(vsp.origin()+vsp.num_elements(), X.extensions());
      copy_n_cast(make_device_ptr(X.origin()),X.num_elements(),Xsp.origin());
      if( std::abs(c-0.0) > 1e-6 )
        copy_n_cast(make_device_ptr(v.origin()),v.num_elements(),vsp.origin());
      ma::product(SPValueType(a),Likn,Xsp,SPValueType(c),vsp);
      copy_n_cast(vsp.origin(),vsp.num_elements(),make_device_ptr(v.origin()));
#else
      ma::product(SPValueType(a),Likn,X,SPValueType(c),v);
#endif
    }

    template<class MatA, class MatB,
             typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality==1)>,
             typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality==1)>,
             typename = void
            >
    void vbias(const MatA& G, MatB&& v, double a=1., double c=0., int k=0) {
      if(walker_type==CLOSED) a*=2.0;
      if(haj.size(0) == 1) {
        if(walker_type==COLLINEAR) {
          int NMO, nel[2];
          NMO = Lnak[0].size(2);
          nel[0] = Lnak[0].size(1);
          nel[1] = Lnak[1].size(1);
          double c_[2];
          c_[0] = c;
          c_[1] = c;
          if( std::abs(c-0.0) < 1e-8 ) c_[1] = 1.0;
          size_t mem_needs = (nel[0]*NMO)+v.num_elements();
          set_buffer(mem_needs);
          SpCVector_ref vsp(TBuff.origin(), v.extensions());
          if( std::abs(c-0.0) > 1e-6 )
            copy_n_cast(make_device_ptr(v.origin()),v.num_elements(),vsp.origin());
          for(int ispin=0, is0=0; ispin<2; ispin++) {
            assert( Lnak[ispin].size(0) == v.size(0) );
            assert( Lnak[ispin].size(1)*Lnak[ispin].size(2) == G.size(0) );
            SpCMatrix_ref Ln(make_device_ptr(Lnak[ispin].origin()), {local_nCV,nel[ispin]*NMO});
#if MIXED_PRECISION
            SpCVector_ref Gsp(vsp.origin()+vsp.num_elements(), {nel[ispin]*NMO});
            copy_n_cast(make_device_ptr(G.origin())+is0,Gsp.num_elements(),Gsp.origin());
            ma::product(SPComplexType(a),Ln,Gsp,SPComplexType(c_[ispin]),vsp);
#else
            ma::product(SPComplexType(a),Ln,G.sliced(is0,is0+nel[ispin]*NMO),SPComplexType(c_[ispin]),v);
#endif
            is0 += nel[ispin]*NMO;
          }
#if MIXED_PRECISION
          copy_n_cast(vsp.origin(),vsp.num_elements(),make_device_ptr(v.origin()));
#endif
        } else {
          assert( Lnak[0].size(1)*Lnak[0].size(2) == G.size(0) );
          assert( Lnak[0].size(0) == v.size(0) );
          SpCMatrix_ref Ln(make_device_ptr(Lnak[0].origin()), {local_nCV,Lnak[0].size(1)*Lnak[0].size(2)});
#if MIXED_PRECISION
          size_t mem_needs = G.num_elements()+v.num_elements();
          set_buffer(mem_needs);
          SpCVector_ref vsp(TBuff.origin(), v.extensions());
          SpCVector_ref Gsp(vsp.origin()+vsp.num_elements(), G.extensions());
          copy_n_cast(make_device_ptr(G.origin()),G.num_elements(),Gsp.origin());
          if( std::abs(c-0.0) > 1e-6 )
            copy_n_cast(make_device_ptr(v.origin()),v.num_elements(),vsp.origin());
          ma::product(SPComplexType(a),Ln,Gsp,SPComplexType(c),vsp);
          copy_n_cast(vsp.origin(),vsp.num_elements(),make_device_ptr(v.origin()));
#else
          ma::product(SPComplexType(a),Ln,G,SPComplexType(c),v);
#endif
        }
      } else {
        // multideterminant is not half-rotated, so use Likn
        assert( Likn.size(0) == G.size(0) );
        assert( Likn.size(1) == v.size(0) );

#if MIXED_PRECISION
        size_t mem_needs = G.num_elements()+v.num_elements();
        set_buffer(mem_needs);
        SpCVector_ref vsp(TBuff.origin(), v.extensions());
        SpCVector_ref Gsp(vsp.origin()+vsp.num_elements(), G.extensions());
        copy_n_cast(make_device_ptr(G.origin()),G.num_elements(),Gsp.origin());
        ma::product(SPValueType(a),ma::T(Likn),Gsp,SPValueType(c),vsp);
        copy_n_cast(vsp.origin(),vsp.num_elements(),make_device_ptr(v.origin()));
#else
        ma::product(SPValueType(a),ma::T(Likn),G,SPValueType(c),v);
#endif
      }
    }

    // v(n,w) = sum_ak L(ak,n) G(w,ak)
    template<class MatA, class MatB,
             typename = typename std::enable_if_t<(std::decay<MatA>::type::dimensionality==2)>,
             typename = typename std::enable_if_t<(std::decay<MatB>::type::dimensionality==2)>
            >
    void vbias(const MatA& G, MatB&& v, double a=1., double c=0., int k=0) {
      if(walker_type==CLOSED) a*=2.0;
      if(haj.size(0) == 1) {
        int nwalk = v.size(1);
        if(walker_type==COLLINEAR) { 
          assert( G.size(1) == v.size(1) );
          int NMO, nel[2];
          NMO = Lnak[0].size(2);
          nel[0] = Lnak[0].size(1);
          nel[1] = Lnak[1].size(1);
          double c_[2];
          c_[0] = c;
          c_[1] = c;  
          if( std::abs(c-0.0) < 1e-8 ) c_[1] = 1.0; 
          size_t mem_needs = (nwalk*nel[0]*NMO)+v.num_elements();
          set_buffer(mem_needs);
          SpCMatrix_ref vsp(TBuff.origin(), v.extensions());
          if( std::abs(c-0.0) > 1e-6 )
            copy_n_cast(make_device_ptr(v.origin()),v.num_elements(),vsp.origin());
          for(int ispin=0, is0=0; ispin<2; ispin++) {
            assert( Lnak[ispin].size(0) == v.size(0) );
            assert( Lnak[ispin].size(1) == G.size(0) );
            SpCMatrix_ref Ln(make_device_ptr(Lnak[ispin].origin()), {local_nCV,nel[ispin]*NMO});
#if MIXED_PRECISION
            SpCMatrix_ref Gsp(vsp.origin()+vsp.num_elements(), {nel[ispin]*NMO,nwalk});
            copy_n_cast(make_device_ptr(G.origin())+is0*nwalk,Gsp.num_elements(),Gsp.origin());
            ma::product(SPComplexType(a),Ln,Gsp,SPComplexType(c_[ispin]),vsp);
#else
            ma::product(SPComplexType(a),Ln,G.sliced(is0,is0+nel[ispin]*NMO),SPComplexType(c_[ispin]),v);
#endif
            is0 += nel[ispin]*NMO;
          }
#if MIXED_PRECISION
          copy_n_cast(vsp.origin(),vsp.num_elements(),make_device_ptr(v.origin()));
#endif
        } else {
          assert( G.size(0) == v.size(1) );
          assert( Lnak[0].size(1)*Lnak[0].size(2) == G.size(1) );
          assert( Lnak[0].size(0) == v.size(0) );
          SpCMatrix_ref Ln(make_device_ptr(Lnak[0].origin()), {local_nCV,Lnak[0].size(1)*Lnak[0].size(2)});
#if MIXED_PRECISION
          size_t mem_needs = G.num_elements()+v.num_elements();
          set_buffer(mem_needs);
          SpCMatrix_ref vsp(TBuff.origin(), v.extensions());
          SpCMatrix_ref Gsp(vsp.origin()+vsp.num_elements(), G.extensions());
          copy_n_cast(make_device_ptr(G.origin()),G.num_elements(),Gsp.origin());
          if( std::abs(c-0.0) > 1e-6 )
            copy_n_cast(make_device_ptr(v.origin()),v.num_elements(),vsp.origin());
          ma::product(SPComplexType(a),Ln,ma::T(Gsp),SPComplexType(c),vsp);
          copy_n_cast(vsp.origin(),vsp.num_elements(),make_device_ptr(v.origin()));
#else
          ma::product(SPComplexType(a),Ln,ma::T(G),SPComplexType(c),v);
#endif
        }
      } else {
        // multideterminant is not half-rotated, so use Likn
        assert( Likn.size(0) == G.size(0) );
        assert( Likn.size(1) == v.size(0) );
        assert( G.size(1) == v.size(1) );

#if MIXED_PRECISION
        size_t mem_needs = G.num_elements()+v.num_elements();
        set_buffer(mem_needs);
        SpCMatrix_ref vsp(TBuff.origin(), v.extensions());
        SpCMatrix_ref Gsp(vsp.origin()+vsp.num_elements(), G.extensions());
        copy_n_cast(make_device_ptr(G.origin()),G.num_elements(),Gsp.origin());
        ma::product(SPValueType(a),ma::T(Likn),Gsp,SPValueType(c),vsp);
        copy_n_cast(vsp.origin(),vsp.num_elements(),make_device_ptr(v.origin()));
#else
        ma::product(SPValueType(a),ma::T(Likn),G,SPValueType(c),v);
#endif
      }
    }

    template<class Mat, class MatB>
    void generalizedFockMatrix(Mat&& G, MatB&& Fp, MatB&& Fm)
    {
      //int nwalk = G.size(0);
      //int nspin = (walker_type==COLLINEAR?2:1);
      //int NMO = hij.size(0);
      //int nel[2];
      //assert(Fp.size(0)==nwalk);
      //assert(Fm.size(0)==nwalk);
      //assert(G[0].num_elements() == nspin*NMO*NMO);
      //assert(Fp[0].num_elements() == nspin*NMO*NMO);
      //assert(Fm[0].num_elements() == nspin*NMO*NMO);

      //// Rwn[nwalk][nCV]: 1+nspin copies  
      //// Twpqn[nwalk][NMO][NMO][nCV]: 1+nspin copies
      //// extra copies 

      //long LBytes = std::max(max_memory_MB*1024L*1024L/long(sizeof(SPComplexType)),TBuff.num_elements());
//#if MIXED_PRECISION
      //LBytes -= long((3*nspin+1)*nwalk*NMO*NMO);   // G, Fp, Fm and Gt
//#else
      //LBytes -= long((1+nspin)*nwalk*NMO*NMO);  //  G and Gt
//#endif
      //LBytes *= long(sizeof(SPComplexType));  
      //int Bytes = int(LBytes / long(2*(NMO*NMO+1)*local_nCV*sizeof(SPComplexType)));
      //int nwmax = std::min( std::max(1, Bytes), nwalk);
      //assert(nwmax >= 1 && nwmax <= nwmax);
//#if MIXED_PRECISION
      //size_t mem_needs = size_t(nwmax*2*(NMO*NMO+1)*local_nCV + (3*nspin+1)*nwalk*NMO*NMO);
//#else
      //size_t mem_needs = size_t(nwmax*2*(NMO*NMO+1)*local_nCV + (nspin+1)*nwalk*NMO*NMO);
//#endif
      //size_t cnt(0);
      //set_buffer(mem_needs);

//#if MIXED_PRECISION
      //SpCMatrix_ref Fp_(TBuff.origin(),{nwalk,nspin*NMO*NMO});
      //cnt += Fp_.num_elements();
      //SpCMatrix_ref Fm_(TBuff.origin(),{nwalk,nspin*NMO*NMO});
      //cnt += Fm_.num_elements();
      //fill_n(TBuff.origin(),cnt,SPComplexType(0.0));
//#else
      //SpCMatrix_ref Fp_(make_device_ptr(Fp.origin()),{nwalk,nspin*NMO*NMO});
      //SpCMatrix_ref Fm_(make_device_ptr(Fm.origin()),{nwalk,nspin*NMO*NMO});
      //fill_n(make_device_ptr(Fp.origin()),Fp.num_elements(),ComplexType(0.0));
      //fill_n(make_device_ptr(Fm.origin()),Fp.num_elements(),ComplexType(0.0));
//#endif

      //SPComplexType scl = (walker_type==CLOSED?2.0:1.0);
      //std::vector<sp_pointer> Aarray;
      //std::vector<sp_pointer> Barray;
      //std::vector<sp_pointer> Carray;
      //Aarray.reserve(nwalk);
      //Barray.reserve(nwalk);
      //Carray.reserve(nwalk);

      //int nw0(0);
      //while(nw0 < nwalk) {

        //int nw = std::min(nwalk-nw0,nwmax);
        //size_t cnt_(cnt);

        //sp_pointer ptr(nullptr);
        //// transpose/cast G
//#if MIXED_PRECISION
        //ptr = TBuff.origin()+cnt_;
        //cnt_ += nspin*nw*NMO*NMO;
        //for(int ispin=0, is0=0, ip=0; ispin<nspin; ispin++, is0+=NMO*NMO) 
          //for(int n=0; n<nw; ++n, ip+=NMO*NMO) 
            //copy_n_cast(make_device_ptr(G[nw0+n].origin())+is0,NMO*NMO,ptr+ip);
//#else
        //if(nspin==1) {
          //ptr = make_device_ptr(G[nw0].origin());
        //} else {
          //ptr = TBuff.origin()+cnt_;
          //cnt_ += nspin*nw*NMO*NMO;
          //using std::copy_n;
          //for(int ispin=0, is0=0, ip=0; ispin<nspin; ispin++, is0+=NMO*NMO) 
            //for(int n=0; n<nw; ++n, ip+=NMO*NMO) 
              //copy_n(make_device_ptr(G[nw0+n].origin())+is0,NMO*NMO,ptr+ip);
        //}
//#endif
        //SpCTensor_ref GF(ptr,{nspin,nw*NMO,NMO});   // now contains G in the correct structure [spin][w][i][j]
        //SpCMatrix_ref Gt(TBuff.origin()+cnt_,{NMO*NMO,nw});  // reserved space for G transposed
        //cnt_ += Gt.num_elements();
        //fill_n(Gt.origin(),Gt.num_elements(),SPComplexType(0.0));

        //SpCMatrix_ref Rnw(TBuff.origin()+cnt_,{local_nCV,nw});
        //cnt_ += Rnw.num_elements(); 
        //// calculate Rwn
        //for(int ispin=0; ispin<nspin; ispin++) {
          //SpCMatrix_ref G_(GF[ispin].origin(),{nw,NMO*NMO});
          //ma::add(SPComplexType(1.0),Gt,SPComplexType(1.0),ma::T(G_),Gt);
        //}
        //// R[n,w] = \sum_ik L[n,ik] G[ik,w]
        //ma::product(SPValueType(1.0),ma::T(Likn),Gt,SPValueType(0.0),Rnw);
        //SpCMatrix_ref Rwn(TBuff.origin()+cnt_,{nw,local_nCV});
        //cnt_ += Rwn.num_elements();
        //ma::transpose(Rnw,Rwn);

        //// add coulomb contribution of <pr||qs>Grs term to Fp, reuse Gt for temporary storage
        //// Fp[p,t] = \sum_{jl} L[p,t,n] L[j,l,n] P[j,l]
        //// Fp[pt,w] = \sum_n L[pt,n] R[n,w]
        //ma::product(SPValueType(1.0),Likn,Rnw,SPValueType(0.0),Gt);
        //for(int ispin=0; ispin<nspin; ispin++) {
          //ma::add(SPComplexType(1.0),Fp_({nw0,nw0+nw},{ispin*NMO*NMO,(ispin+1)*NMO*NMO}),
                  //SPComplexType(scl),ma::T(Gt),Fp_({nw0,nw0+nw},{ispin*NMO*NMO,(ispin+1)*NMO*NMO}));
        //}

        //// L[i,kn]
        //SpRMatrix_ref Ln(make_device_ptr(Likn.origin()),{NMO,NMO*local_nCV});
        //// T[w,p,t,n] = \sum_{l} L[l,t,n] P[w,l,p]
        //SpCMatrix_ref Twptn(TBuff.origin()+cnt_,{nw*NMO,NMO*local_nCV});
        //cnt_ += Twptn.num_elements();
        //// transpose for faster contraction
        //SpCMatrix_ref Taux(TBuff.origin()+cnt_,{nw*NMO,NMO*local_nCV});
        //cnt_ += Taux.num_elements();
        //SpCTensor_ref Twptn3D(Twptn.origin(),{nw,NMO,NMO*local_nCV});
        //SpCMatrix_ref Ttnwp(Taux.origin(),{NMO*local_nCV,nw*NMO});
        //SpCMatrix_ref Gt_(Gt.origin(),{NMO,nw*NMO});

        //for(int ispin=0, is0=0; ispin<nspin; ispin++, is0+=NMO*NMO) {

          //SpCMatrix_ref G_(GF[ispin].origin(),{nw*NMO,NMO});
          //ma::transpose(G_,Gt_);

          //// J = \sum_{iklr} L[i,k,n] L[q,l,n] P[s,p,l] P[r,i,k]
          //// R[n] = \sum_{ik} L[i,k,n] P[r,i,k]
          //// Here T[tn,wp] = \sum_{l} L[tn,l] P[l,wp]
          //ma::product(SPValueType(1.0),ma::T(Ln),Gt_,SPValueType(0.0),Ttnwp);
          //// T[wp,tn]
          //ma::transpose(Ttnwp,Twptn);

          //// transpose Twptn -> Twtpn=Taux
          //using ma::transpose_wabn_to_wban;
          //// T[wt,pn]
          //transpose_wabn_to_wban(nw,NMO,NMO,local_nCV,Twptn.origin(),Taux.origin());

          //// add exchange component to Fm_
          //Aarray.clear();
          //Barray.clear();
          //Carray.clear();
          //for(int w=0; w<nw; w++) {
            //Aarray.push_back(Taux[w].origin());
            //Barray.push_back(Twptn3D[w].origin());
            //Carray.push_back(Fm_[w].origin()+is0);
          //}
          //using ma::gemmBatched;
          //// careful with expected Fortran ordering here!!!
          //// K[p,q] = \sum_{ln} T[n,l,p] T[n,q,l]
          ////          \sum_{ln} T[nl,p] T[nl,q]
          //gemmBatched('T','N',NMO,NMO,NMO*local_nCV,
                      //SPComplexType(1.0),Aarray.data(),NMO*local_nCV,
                      //Barray.data(),NMO*local_nCV,
                      //SPComplexType(1.0),Carray.data(),NMO,nw);

          //// add coulomb component to Fm_
          //Aarray.clear();
          //Barray.clear();
          //Carray.clear();
          //for(int w=0; w<nw; w++) {
            //Aarray.push_back(Twptn3D[w].origin());
            //Barray.push_back(Rwn[w].origin());
            //Carray.push_back(Fm_[w].origin()+is0);
          //}
          //using ma::gemmBatched;
          //// careful with expected Fortran ordering here!!!
          //// J[w][pq] = \sum_{n} T[w][pq,n] R[w][n]
          //gemmBatched('T','N',NMO*NMO,1,local_nCV,
                      //SPComplexType(-1.0)*scl,Aarray.data(),local_nCV,
                      //Barray.data(),local_nCV,
                      //SPComplexType(1.0),Carray.data(),NMO*NMO,nw);

          //// Fp
          //// add coulomb component to Fp_, same as Fm_ above
          //Aarray.clear();
          //Barray.clear();
          //Carray.clear();
          //for(int w=0; w<nw; w++) {
            //Aarray.push_back(Twptn3D[w].origin());
            //Barray.push_back(Rwn[w].origin());
            //Carray.push_back(Fp_[w].origin()+is0);
          //}
          //using ma::gemmBatched;
          //// careful with expected Fortran ordering here!!!
          //// Coulomb component
          //gemmBatched('T','N',NMO*NMO,1,local_nCV,
                      //SPComplexType(-1.0)*scl,Aarray.data(),local_nCV,
                      //Barray.data(),local_nCV,
                      //SPComplexType(1.0),Carray.data(),NMO*NMO,nw);

          //// add exchange component of Fp_
          //Aarray.clear();
          //Barray.clear();
          //Carray.clear();
          //for(int w=0; w<nw; w++) {
            //Aarray.push_back(Taux[w].origin());
            //Barray.push_back(Twptn3D[w].origin());
            //Carray.push_back(Fp_[w].origin()+is0);

            //// add exchange contribution of <pr||qs>Grs term by adding Lptn to Twptn
            //// dispatch directly from here to be able to add to the real part only
            //// K1B[p,q] = -\sum_{jl} L[jt,n] L[pl,n] P[jl]
            //using ma::axpy;
            //axpy(Likn.num_elements(), SPRealType(-1.0),
                    //ma::pointer_dispatch(Likn.origin()), 1,
                    //pointer_cast<SPRealType>(ma::pointer_dispatch(Twptn[w].origin())), 2);
          //}
          //using ma::gemmBatched;
          //// careful with expected Fortran ordering here!!!
          //gemmBatched('T','N',NMO,NMO,NMO*local_nCV,
                      //SPComplexType(1.0),Aarray.data(),NMO*local_nCV,
                      //Barray.data(),NMO*local_nCV,
                      //SPComplexType(1.0),Carray.data(),NMO,nw);

        //} // ispin

        //nw0 += nw;
      //}

//#if MIXED_PRECISION
      //copy_n_cast(Fp_.origin(),Fp_.num_elements(),make_device_ptr(Fp.origin()));
      //copy_n_cast(Fm_.origin(),Fm_.num_elements(),make_device_ptr(Fm.origin()));
//#endif

      //// add one body terms now
      //{

        //std::vector<pointer> Aarr;
        //std::vector<pointer> Barr;
        //std::vector<pointer> Carr;
        //Aarr.reserve(nspin*nwalk);
        //Barr.reserve(nspin*nwalk);
        //Carr.reserve(nspin*nwalk);
        //// Fm -= G[w][p][r] * h[q][r]
        //Aarr.clear();
        //Barr.clear();
        //Carr.clear();
        //for(int ispin=0, is0=0; ispin<nspin; ispin++, is0+=NMO*NMO) {
          //for(int w=0; w<nwalk; w++) {
            //Aarr.push_back(hij_dev.origin());
            //Barr.push_back(G[w].origin()+is0);
            //Carr.push_back(Fm[w].origin()+is0);
          //}
        //}
        //using ma::gemmBatched;
        //// careful with expected Fortran ordering here!!!
        //gemmBatched('T','N',NMO,NMO,NMO,
                    //ComplexType(-1.0),Aarr.data(),NMO,
                    //Barr.data(),NMO,
                    //ComplexType(1.0),Carr.data(),NMO,Aarr.size());


        //// Fp -= G[w][r][p] * h[q][r]
        //Aarr.clear();
        //Barr.clear();
        //Carr.clear();
        //C4Tensor_ref Fp4D(make_device_ptr(Fp.origin()),{nwalk,nspin,NMO,NMO});
        //for(int ispin=0, is0=0; ispin<nspin; ispin++, is0+=NMO*NMO) {
          //for(int w=0; w<nwalk; w++) {
            //Aarr.push_back(hij_dev.origin());
            //Barr.push_back(G[w].origin()+is0);
            //Carr.push_back(Fp[w].origin()+is0);

            //// add diagonal contribution to Fp
            //ma::add(ComplexType(1.0),Fp4D[w][ispin],
                    //ComplexType(1.0),ma::T(hij_dev),
                    //Fp4D[w][ispin]);
          //}
        //}
        //using ma::gemmBatched;
        //// careful with expected Fortran ordering here!!!
        //gemmBatched('T','T',NMO,NMO,NMO,
                    //ComplexType(-1.0),Aarr.data(),NMO,
                    //Barr.data(),NMO,
                    //ComplexType(1.0),Carr.data(),NMO,Aarr.size());

      //}

    }

    bool distribution_over_cholesky_vectors() const{ return true; }
    int number_of_ke_vectors() const{ return local_nCV; }
    int local_number_of_cholesky_vectors() const{ return local_nCV; }
    int global_number_of_cholesky_vectors() const{ return global_nCV; }
    int global_origin_cholesky_vector() const{ return global_origin; }

    // transpose=true means G[nwalk][ik], false means G[ik][nwalk]
    bool transposed_G_for_vbias() const{ 
        return ((haj.size(0) == 1) && (walker_type!=COLLINEAR)); 
    } 
    bool transposed_G_for_E() const{return true;}
    // transpose=true means vHS[nwalk][ik], false means vHS[ik][nwalk]
    bool transposed_vHS() const{return false;}

    bool fast_ph_energy() const { return false; }

    boost::multi::array<ComplexType,2> getHSPotentials()
    {
      return boost::multi::array<ComplexType,2>{};
    }

  private:

    Allocator allocator_;
    SpAllocator sp_allocator_;

    WALKER_TYPES walker_type;

    long max_memory_MB;
    int global_origin;
    int global_nCV;
    int local_nCV;

    ValueType E0;

    // bare one body hamiltonian
    mpi3RMatrix hij;

    // one body hamiltonian
    shmCMatrix hij_dev;

    // (potentially half rotated) one body hamiltonian
    shmCMatrix haj;

    //Cholesky Tensor Lik[i][k][n]
    shmSpRMatrix Likn;

    // permuted half-tranformed Cholesky tensor
    // Lnak[ 2*idet + ispin ]
    std::vector<shmSpC3Tensor> Lnak;

    // one-body piece of Hamiltonian factorization
    mpi3CMatrix vn0;

    // shared buffer space
    // using matrix since there are issues with vectors
    SpVector TBuff;

    myTimer Timer;

    void set_buffer(size_t N) {
      if(TBuff.num_elements() < N) {
        app_log()<<" Resizing buffer space in Real3IndexFactorization_batched_v2 to " <<N*sizeof(SPComplexType)/1024.0/1024.0 <<" MBs. \n";
        {
          TBuff = std::move(SpVector(iextensions<1u>{N}));
        }
        memory_report();
        using std::fill_n;
        fill_n(TBuff.origin(),N,SPComplexType(0.0));
      }
    }

};

}

}

#endif
