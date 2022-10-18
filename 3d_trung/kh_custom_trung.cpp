//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file kh.cpp
//! \brief Problem generator for KH instability.
//!

// C headers

// C++ headers
#include <algorithm>  // min, max
#include <cmath>      // log
#include <cstring>    // strcmp()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../defs.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp" // diffusion
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"
#include "../utils/utils.hpp"

// User Defined Functions
void ConstantConduction(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke);
void ConstantViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke);
void SpitzerConduction(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke);
void SpitzerViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke);

// User Defined Boundary Conditions
void ConstantShearInflowOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void ConstantShearInflowInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void ConstantShearInflowOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void ConstantShearInflowInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int il, int iu, int jl, int ju, int kl, int ku, int ngh);

// Global variables put in unnamed namespace to avoid linkage issues
namespace {
  int iprob;
  Real gamma_adi;
  Real rho_0, pgas_0;
  Real vel_shear, vel_pert;
  Real smoothing_thickness, smoothing_thickness_vel;
  Real z_max, z_min;
  Real z_top, z_bot;
  Real density_contrast;
  Real lambda_pert;
  Real T_cond_max;
  Real visc_factor;
} // namespace


//----------------------------------------------------------------------------------------
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Read Problem Parameters
  iprob                  = pin->GetOrAddInteger("problem","iprob",1); //problem identifier, since KH file has 3 separate problems that can be run
  gamma_adi              = pin->GetReal("hydro",   "gamma"); //adiabatic gamma factor
  rho_0                  = pin->GetReal("problem", "rho_0"); //density of gas on the sides
  pgas_0                 = pin->GetReal("problem", "pgas_0"); //pressure of gas on the sides
  density_contrast       = pin->GetReal("problem", "density_contrast"); //fraction between the density in the middle and the density on the side
  vel_shear              = pin->GetReal("problem", "vel_shear"); //relative shear velocity?


  // used for cooling
  z_min = mesh_size.x3min;
  z_max = mesh_size.x3max;

  // Initial conditions and Boundary values
  smoothing_thickness = pin->GetReal("problem", "smoothing_thickness"); //initialize a gradual change in density at the boundary. smoothing_thickness is the width of the boundary
  smoothing_thickness_vel = pin->GetOrAddReal("problem", "smoothing_thickness_vel", -100.0); //velocity of the boundary layer
  if (smoothing_thickness_vel < 0){
    smoothing_thickness_vel = smoothing_thickness;
  } //why? This seems random
  vel_pert       = pin->GetReal("problem", "vel_pert"); //velocity of the perturbation 
  lambda_pert         = pin->GetReal("problem", "lambda_pert"); //length of the boundary layer?
  z_top               = pin->GetReal("problem", "z_top"); //boundary of the cold gas layer
  z_bot               = pin->GetReal("problem", "z_bot"); //''

  // Boundary Conditions -----------------------------------------------------------------
  bool ConstantShearInflowOuterX2_on = pin->GetOrAddBoolean("problem", "ConstantShearInflowOuterX2_on", false); //indicate if shear is in the 2nd or 3rd axis, probably always want to be true
  bool ConstantShearInflowInnerX2_on = pin->GetOrAddBoolean("problem", "ConstantShearInflowInnerX2_on", false);
  bool ConstantShearInflowOuterX3_on = pin->GetOrAddBoolean("problem", "ConstantShearInflowOuterX3_on", false);
  bool ConstantShearInflowInnerX3_on = pin->GetOrAddBoolean("problem", "ConstantShearInflowInnerX3_on", false);

  // Enroll 3D boundary condition
  if(mesh_bcs[BoundaryFace::inner_x3] == GetBoundaryFlag("user")) {
    if (ConstantShearInflowInnerX3_on) EnrollUserBoundaryFunction(BoundaryFace::inner_x3, ConstantShearInflowInnerX3); 
  }
  if(mesh_bcs[BoundaryFace::outer_x3] == GetBoundaryFlag("user")) {
    if (ConstantShearInflowOuterX3_on) EnrollUserBoundaryFunction(BoundaryFace::outer_x3, ConstantShearInflowOuterX3); //MODIFICATION: added "BoundaryFace::" before "inner_x3" amd "outer_x3"
  }

  // Enroll 2D boundary condition
  if(mesh_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("user")) {
    if (ConstantShearInflowInnerX2_on) EnrollUserBoundaryFunction(BoundaryFace::inner_x2, ConstantShearInflowInnerX2);
  }
  if(mesh_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("user")) {
    if (ConstantShearInflowOuterX2_on) EnrollUserBoundaryFunction(BoundaryFace::outer_x2, ConstantShearInflowOuterX2);
  }

  // Read Microphysics -----------------------------------------------------------------
  bool SpitzerViscosity_on = pin->GetOrAddBoolean("problem", "SpitzerViscosity_on", false);
  bool SpitzerConduction_on = pin->GetOrAddBoolean("problem", "SpitzerConduction_on", false);

  if (SpitzerViscosity_on){
    visc_factor            = pin->GetOrAddReal("problem","visc_factor",1.0);
    T_cond_max             = pin->GetReal("problem","T_cond_max");
    EnrollViscosityCoefficient(SpitzerViscosity);
  }
  if (SpitzerConduction_on){
    EnrollConductionCoefficient(SpitzerConduction);
  }

  bool ConstantViscosity_on = pin->GetOrAddBoolean("problem", "ConstantViscosity_on", false);
  bool ConstantConduction_on = pin->GetOrAddBoolean("problem", "ConstantConduction_on", false);

  if (ConstantViscosity_on){
    EnrollViscosityCoefficient(ConstantViscosity);
  }
  if (ConstantConduction_on){
    EnrollConductionCoefficient(ConstantConduction);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for the Kelvin-Helmholtz test

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  bool noisy_IC = pin->GetOrAddBoolean("problem", "noisy_IC", false); //noisy initial condition? Like initializing a random-ish field??
  std::int64_t iseed = -1 - gid; //what does gid do?

  // Prepare index bounds
  int il = is - NGHOST;
  int iu = ie + NGHOST;
  int jl = js;
  int ju = je;
  if (block_size.nx2 > 1) {
    jl -= (NGHOST);
    ju += (NGHOST);
  }
  int kl = ks;
  int ku = ke;
  if (block_size.nx3 > 1) {
    kl -= (NGHOST);
    ku += (NGHOST);
  }

  if (iprob == 1) {
    // Read problem parameters
    for (int k=kl; k<=ku; k++) {
      Real z = pcoord->x3v(k);
      for (int j=jl; j<=ju; j++) {
        Real y = pcoord->x2v(j);
        Real r = std::sqrt(SQR(z) + SQR(y)); //set up radial step function for a cylinder, r = sqrt(z^2 + y^2)
        for (int i=il; i<=iu; i++) {
          Real x = pcoord->x1v(i);
          // 3D
          if (block_size.nx3 > 1) {
            phydro->u(IDN,k,j,i) = 0.5 * (density_contrast + (rho_0 * (1.0 + (1.0 - density_contrast) * (std::tanh((r-z_top)/smoothing_thickness) ) ) ) );
            phydro->u(IPR,k,j,i) = pgas_0;
            phydro->u(IVX,k,j,i) = 0.5 * vel_shear * std::tanh((z_top-r)/smoothing_thickness);
            phydro->u(IVY,k,j,i) = 0.0;
            phydro->u(IVZ,k,j,i) = 0.0;
            // Perturbations
            if (lambda_pert > 0.0){
              phydro->u(IVZ,k,j,i) = vel_pert;
              phydro->u(IVZ,k,j,i) *= (std::exp(-SQR((r-z_bot)/smoothing_thickness_vel)) + std::exp(-SQR((r-z_top)/smoothing_thickness_vel) ) );
              phydro->u(IVZ,k,j,i) *= std::sin(2*PI*x/lambda_pert) * std::sin(2*PI*r/lambda_pert) ;
            }
          }
          // 2D
          else if (block_size.nx2 > 1) {
            phydro->u(IDN,k,j,i) = rho_0 * (1.0 + (density_contrast-1.0) * 0.5 * ( std::tanh((y-z_bot)/smoothing_thickness) - std::tanh((y-z_top)/smoothing_thickness) ) );
            phydro->u(IM1,k,j,i) = phydro->u(IDN,k,j,i) * vel_shear * ( 0.5 - 0.5 * ( std::tanh((y-z_bot)/smoothing_thickness) - std::tanh((y-z_top)/smoothing_thickness) ));
            phydro->u(IM2,k,j,i) = 0.0;
            // Perturbations
            if (lambda_pert > 0.0){
              phydro->u(IM2,k,j,i) = phydro->u(IDN,k,j,i) * vel_pert;
              phydro->u(IM2,k,j,i) *= (std::exp(-SQR((y-z_bot)/smoothing_thickness_vel)) + std::exp(-SQR((y-z_top)/smoothing_thickness_vel)));
              phydro->u(IM2,k,j,i) *=  std::sin(2*PI*x/lambda_pert);
            } // lambda_pert if
            if (noisy_IC){
              phydro->u(IM2,k,j,i) *= ran2(&iseed); //if the noisy IC is declared, introduce some random motion in the y direction (for 2D, no 3D implementation written yet)
            }
            if (NON_BAROTROPIC_EOS) {
              phydro->u(IEN,k,j,i) = pgas_0/(gamma_adi-1.0) + 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i)))/phydro->u(IDN,k,j,i);
            }
          } // 2D if
        } // i for
      } // j for
    } // k for

    // initialize uniform interface B
    if (MAGNETIC_FIELDS_ENABLED) {
      Real b0 = pin->GetReal("problem","b0");
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=is; i<=ie+1; i++) {
            pfield->b.x1f(k,j,i) = b0;
          }
        }
      }
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je+1; j++) {
          for (int i=is; i<=ie; i++) {
            pfield->b.x2f(k,j,i) = 0.0;
          }
        }
      }
      for (int k=ks; k<=ke+1; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=is; i<=ie; i++) {
            pfield->b.x3f(k,j,i) = 0.0;
          }
        }
      }
      if (NON_BAROTROPIC_EOS) {
        for (int k=ks; k<=ke; k++) {
          for (int j=js; j<=je; j++) {
            for (int i=is; i<=ie; i++) {
              phydro->u(IEN,k,j,i) += 0.5*b0*b0;
            }
          }
        }
      }
    }
  }


  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ConstantShearInflowInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          FaceField &b, Real time, Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief ConstantShearInflow boundary conditions, inner x2 boundary

void ConstantShearInflowInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  // copy hydro variables into ghost zones
    for (int k=kl; k<=ku; k++) {
    for (int j=1; j<=ngh; j++) {
      for (int i=il; i<=iu; i++) {
        for (int n=0; n<(NHYDRO); n++) {
        Real z = pco->x2v(jl-j);
        prim(n,k,jl-j,i) = prim(n,k,jl,i);
        if ( n == IPR ){
          prim(IPR,k,jl-j,i) = pgas_0;
        } 
        if ( n == IDN ){
          prim(IDN,k,jl-j,i) = rho_0 * (1.0 + (density_contrast-1.0) * 0.5 * ( std::tanh((z-z_bot)/smoothing_thickness) - std::tanh((z-z_top)/smoothing_thickness) ) );
        } 
        if ( n == IVX ){
          prim(IVX,k,jl-j,i) = vel_shear * ( 0.5 - 0.5 * ( std::tanh((z-z_bot)/smoothing_thickness) - std::tanh((z-z_top)/smoothing_thickness) ));
        } 
      }
    }}
  }

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu+1; ++i) {
        b.x1f(k,(jl-j),i) = b.x1f(k,jl,i);
      }
    }}

    for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu; ++i) {
        b.x2f(k,(jl-j),i) = b.x2f(k,jl,i);
      }
    }}

    for (int k=kl; k<=ku+1; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu; ++i) {
        b.x3f(k,(jl-j),i) = b.x3f(k,jl,i);
      }
    }}
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ConstantShearInflowOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          FaceField &b, Real time, Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief ConstantShearInflow boundary conditions, outer x2 boundary

void ConstantShearInflowOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  // copy hydro variables into ghost zones
    for (int k=kl; k<=ku; k++) {
    for (int j=1; j<=ngh; j++) {
#pragma omp simd
      for (int i=il; i<=iu; i++) {
        for (int n=0; n<(NHYDRO); n++) {
        Real z = pco->x2v(ju+j);
        prim(n,k,ju+j,i) = prim(n,k,ju,i);
        if ( n == IPR ){
          prim(IPR,k,ju+j,i) = pgas_0;
        } 
        if ( n == IDN ){
          prim(IDN,k,ju+j,i) = rho_0 * (1.0 + (density_contrast-1.0) * 0.5 * ( std::tanh((z-z_bot)/smoothing_thickness) - std::tanh((z-z_top)/smoothing_thickness) ) );
        } 
        if ( n == IVX ){
          prim(IVX,k,ju+j,i) = vel_shear * ( 0.5 - 0.5 * ( std::tanh((z-z_bot)/smoothing_thickness) - std::tanh((z-z_top)/smoothing_thickness) ));
        } 
      }
    }}
  }

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu+1; ++i) {
        b.x1f(k,(ju+j  ),i) = b.x1f(k,(ju  ),i);
      }
    }}

    for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu; ++i) {
        b.x2f(k,(ju+j+1),i) = b.x2f(k,(ju+1),i);
      }
    }}

    for (int k=kl; k<=ku+1; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu; ++i) {
        b.x3f(k,(ju+j  ),i) = b.x3f(k,(ju  ),i);
      }
    }}
  }


  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ConstantShearInflowInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          FaceField &b, Real time, Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief ConstantShearInflow boundary conditions, inner x3 boundary

void ConstantShearInflowInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  // copy hydro variables into ghost zones
  for (int n=0; n<(NHYDRO); n++) {
    for (int k=1; k<=ngh; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {
        Real z = pco->x3v(kl-k);
        prim(n,kl-k,j,i) = prim(n,kl,j,i);
        if ( n == IPR ){
          prim(IPR,kl-k,j,i) = pgas_0;
        } 
        if ( n == IDN ){
          prim(IDN,kl-k,j,i) = rho_0 * (1.0 + (density_contrast-1.0) * 0.5 * ( std::tanh((z-z_bot)/smoothing_thickness) - std::tanh((z-z_top)/smoothing_thickness) ) );
        } 
        if ( n == IVX ){
          prim(IVX,kl-k,j,i) = vel_shear * ( 0.5 - 0.5 * ( std::tanh((z-z_bot)/smoothing_thickness) - std::tanh((z-z_top)/smoothing_thickness) ));
        } 
      }
    }}
  }

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu+1; ++i) {
        b.x1f((kl-k),j,i) = b.x1f(kl,j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju+1; ++j) {
      for (int i=il; i<=iu; ++i) {
        b.x2f((kl-k),j,i) = b.x2f(kl,j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        b.x3f((kl-k),j,i) = b.x3f(kl,j,i);
      }
    }}
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ConstantShearInflowOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          FaceField &b, Real time, Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief ConstantShearInflow boundary conditions, outer x3 boundary

void ConstantShearInflowOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  // copy hydro variables into ghost zones
  for (int n=0; n<(NHYDRO); n++) {
    for (int k=1; k<=ngh; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {
        Real z = pco->x3v(ku+k);
        prim(n,ku+k,j,i) = prim(n,ku,j,i);
        if ( n == IPR ){
          prim(IPR,ku+k,j,i) = pgas_0;
        } 
        if ( n == IDN ){
          prim(IDN,ku+k,j,i) = rho_0 * (1.0 + (density_contrast-1.0) * 0.5 * ( std::tanh((z-z_bot)/smoothing_thickness) - std::tanh((z-z_top)/smoothing_thickness) ) );
        } 
        if ( n == IVX ){
          prim(IVX,ku+k,j,i) = vel_shear * ( 0.5 - 0.5 * ( std::tanh((z-z_bot)/smoothing_thickness) - std::tanh((z-z_top)/smoothing_thickness) ));
        } 
      }
    }}
  }

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu+1; ++i) {
        b.x1f((ku+k  ),j,i) = b.x1f((ku  ),j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        b.x2f((ku+k  ),j,i) = b.x2f((ku  ),j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        b.x3f((ku+k+1),j,i) = b.x3f((ku+1),j,i);
      }
    }}
  }

  return;
}

// ----------------------------------------------------------------------------------------
// SpitzerViscosity 
// 
void SpitzerViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke) 
{
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is+1; i<=ie; ++i) {
        Real T = prim(IPR,k,j,i)/prim(IDN,k,j,i);
        Real Tpow = T > T_cond_max ? pow(T,2.5) : pow(T_cond_max,2.5);
        phdif->nu(HydroDiffusion::DiffProcess::iso,k,j,i) = visc_factor * phdif->nu_iso/prim(IDN,k,j,i) * Tpow;
      }
    }
  }
  return;
}

// ----------------------------------------------------------------------------------------
// SpitzerConduction 
// 
void SpitzerConduction(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke) 
{
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is+1; i<=ie; ++i) {
        Real T = prim(IPR,k,j,i)/prim(IDN,k,j,i);
        Real Tpow = T > T_cond_max ? pow(T_cond_max,2.5) : pow(T,2.5);
        phdif->kappa(HydroDiffusion::DiffProcess::iso,k,j,i) = phdif->kappa_iso/prim(IDN,k,j,i) * Tpow;
      }
    }
  }
  return;
}

// ----------------------------------------------------------------------------------------
// ConstantViscosity 
// 
void ConstantViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke) 
{
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is+1; i<=ie; ++i) {
        phdif->nu(HydroDiffusion::DiffProcess::iso,k,j,i) = phdif->nu_iso/prim(IDN,k,j,i);
      }
    }
  }
  return;
}

// ----------------------------------------------------------------------------------------
// ConstantConduction 
// 
void ConstantConduction(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke) 
{
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is+1; i<=ie; ++i) {
        phdif->kappa(HydroDiffusion::DiffProcess::iso,k,j,i) = phdif->kappa_iso/prim(IDN,k,j,i);
      }
    }
  }
  return;
}
