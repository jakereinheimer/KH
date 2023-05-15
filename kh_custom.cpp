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
static Real Lambda_cool(Real Temp);
void Cooling(MeshBlock *pmb, const Real time, const Real dt,
             const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
             const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
             AthenaArray<Real> &cons_scalar);

// User Defined Boundary Conditions
void ConstantShearInflowOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void ConstantShearInflowInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
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
  Real radius;
  Real density_contrast;
  Real lambda_pert;
  Real T_cond_max;
  Real visc_factor;
  Real cooling_factor;
  Real T_cold,T_hot;
} // namespace


//----------------------------------------------------------------------------------------
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Read Problem Parameters
  iprob                  = pin->GetOrAddInteger("problem","iprob",1);
  gamma_adi              = pin->GetReal("hydro",   "gamma");
  rho_0                  = pin->GetReal("problem", "rho_0");
  pgas_0                 = pin->GetReal("problem", "pgas_0");
  density_contrast       = pin->GetReal("problem", "density_contrast");
  vel_shear              = pin->GetReal("problem", "vel_shear");

  Real rho_hot = rho_0;
  Real rho_cold = rho_hot * density_contrast;
  T_hot = pgas_0 / rho_hot;
  T_cold = pgas_0 / rho_cold;


  // used for cooling
  z_min = mesh_size.x3min;
  z_max = mesh_size.x3max;

  // Initial conditions and Boundary values
  smoothing_thickness = pin->GetReal("problem", "smoothing_thickness");
  smoothing_thickness_vel = pin->GetOrAddReal("problem", "smoothing_thickness_vel", -100.0);
  if (smoothing_thickness_vel < 0){
    smoothing_thickness_vel = smoothing_thickness;
  }
  vel_pert       = pin->GetReal("problem", "vel_pert");
  lambda_pert         = pin->GetReal("problem", "lambda_pert");
  radius               = pin->GetReal("problem", "radius");

  // Boundary Conditions -----------------------------------------------------------------
  bool ConstantShearInflowOuterX1_on = pin->GetOrAddBoolean("problem", "ConstantShearInflowOuterX1_on", false);
  bool ConstantShearInflowInnerX1_on = pin->GetOrAddBoolean("problem", "ConstantShearInflowInnerX1_on", false);


  // Enroll 2D boundary condition
  if(mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    if (ConstantShearInflowInnerX1_on) EnrollUserBoundaryFunction(BoundaryFace::inner_x1, ConstantShearInflowInnerX1);
  }
  if(mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    if (ConstantShearInflowOuterX1_on) EnrollUserBoundaryFunction(BoundaryFace::outer_x1, ConstantShearInflowOuterX1);
  }

  // Read Microphysics -----------------------------------------------------------------
  bool Cooling_on          = pin->GetOrAddBoolean("problem", "Cooling_on", false);
  if (Cooling_on){
    cooling_factor         = pin->GetOrAddReal("problem", "cooling_factor", 1.0);
    EnrollUserExplicitSourceFunction(Cooling);
  }

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
  bool noisy_IC = pin->GetOrAddBoolean("problem", "noisy_IC", false);
  std::int64_t iseed = -1 - gid;

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
        for (int i=il; i<=iu; i++) {
          Real x = pcoord->x1v(i);
          // 3D
          if (block_size.nx3 > 1) {
            Real r = std::sqrt(SQR(z) + SQR(y));
            Real density= rho_0 * ((density_contrast/2) + 0.5 + (density_contrast-1.0) * 0.5 * -std::tanh((r-radius)/smoothing_thickness) ) ;

            phydro->u(IDN,k,j,i) = density;
            phydro->u(IEN,k,j,i) = pgas_0/(gamma_adi-1);
            phydro->u(IM1,k,j,i) = density * vel_shear * -1 * (std::tanh((r-radius)/smoothing_thickness) );
            phydro->u(IM2,k,j,i) = 0.0;
            phydro->u(IM3,k,j,i) = 0.0;

            // Perturbations
            // A full circular velocity pertubation
            if (lambda_pert >= 0.0){
              if ((density>1.0) && (density<density_contrast)){ // Adding perturbations only too areas between the densities

                Real mag = density * vel_pert ;// is the magnitude of the pertibation
                mag *= std::exp(-1*SQR(r-radius)/(smoothing_thickness));// is the gaussian to center the perturbation

                if (lambda_pert > 0.0){
                mag *= std::sin(2*PI*x/lambda_pert) ; //multiplies both components by a sin to have it at oscilate along x axis
                }

                if (lambda_pert == 0.0) {
                  Real pert_loc= pin->GetOrAddInteger("problem","pert_loc",-2.5);
                  Real pert_width= pin->GetOrAddInteger("problem","pert_width",smoothing_thickness);

                  mag *= std::exp(-1*(SQR(x+(-1*pert_loc))/pert_width)); //multiplies both components by a sin to have it at oscilate along x axis
                } // end second if

                Real theta = std::atan(y/z); // is the angle the current point is on for calculating the impact for each dimension
                phydro->u(IM3,k,j,i) = mag * std::cos(theta); //z component magnitude with cos of theta
                phydro->u(IM2,k,j,i) = mag * std::sin(theta); //y component magnitude with sin of theta
            } //end density if statement
            } // end pert

            // Add random noise if noise wants to be added
            if (noisy_IC){
              phydro->u(IM2,k,j,i) *= ran2(&iseed); 
              phydro->u(IM3,k,j,i) *= ran2(&iseed); 
            }

            // sets pressure based on velocity
            if (NON_BAROTROPIC_EOS) {
              phydro->u(IEN,k,j,i) = pgas_0/(gamma_adi-1.0) + 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i)))/phydro->u(IDN,k,j,i);
            }
          }
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
// calculated edot_cool 
// the cooling curve is set to be a sine wave 
// negative sine is make it cooling at low temp
static Real Lambda_cool(Real Temp)
{
  return std::sin(PI*std::log(Temp/T_cold) / (std::log((T_hot-T_cold)/T_cold))) ;
}

void Cooling(MeshBlock *pmb, const Real time, const Real dt,
             const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
             const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
             AthenaArray<Real> &cons_scalar) {
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        Real temp = prim(IPR,k,j,i) / prim(IDN,k,j,i);
          cons(IEN,k,j,i) -= dt * cooling_factor * prim(IDN,k,j,i) * prim(IDN,k,j,i) * Lambda_cool(temp);
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

void ConstantShearInflowInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  // copy hydro variables into ghost zones
    for (int k=kl; k<=ku; k++) {
    Real z = pco->x3v(k);
    for (int j=jl; j<=ju; j++) {
      Real y = pco->x2v(j);
      for (int i=1; i<=ngh; i++) {
        //Real x = pco->x1v(i);
        Real r = std::sqrt(SQR(z) + SQR(y));
        Real density= rho_0 * ((density_contrast/2) + 0.5 + (density_contrast-1.0) * 0.5 * -std::tanh((r-radius)/smoothing_thickness) ) ;

        for (int n=0; n<(NHYDRO); n++) {
        prim(n,k,j,il-i) = prim(n,k,j,il);
        if ( n == IPR ){
          prim(IPR,k,j,il-i) = pgas_0;
        } 
        if ( n == IDN ){
          prim(IDN,k,j,il-i) = density;
        } 
        if ( n == IVX ){
          prim(IVX,k,j,il-i) = vel_shear * -1 * (std::tanh((r-radius)/smoothing_thickness) );
        }
        }
      if (r>radius){
          prim(IDN,k,j,il-i) = 1e-5;
          prim(IVX,k,j,il-i) = 1e-5;
          prim(IPR,k,j,il-i) = 1e-5;
          }
    }
    }
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

void ConstantShearInflowOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  // copy hydro variables into ghost zones
    for (int k=kl; k<=ku; k++) {
      Real z=pco->x3v(k);
    for (int j=jl; j<=ju; j++) {
      Real y=pco->x2v(j);
#pragma omp simd
      for (int i=1; i<=ngh; i++) {
        //Real x = pco->x3v(i);
        Real r = std::sqrt(SQR(z) + SQR(y));
        Real density= rho_0 * ((density_contrast/2) + 0.5 + (density_contrast-1.0) * 0.5 * -std::tanh((r-radius)/smoothing_thickness) ) ;

        for (int n=0; n<(NHYDRO); n++) {
        prim(n,k,j,iu+i) = prim(n,k,j,iu);
        if ( n == IPR ){
          prim(IPR,k,j,iu+i) = pgas_0;
        } 
        if ( n == IDN ){
          prim(IDN,k,j,iu+i) = density;
        } 
        if ( n == IVX ){
          prim(IVX,k,j,iu+i) = vel_shear * -1 * (std::tanh((r-radius)/smoothing_thickness) );
        }
        }
      if (r<radius){
          prim(IDN,k,j,iu+i) = 1e-5;
          prim(IVX,k,j,iu+i) = 1e-5;
          prim(IPR,k,j,iu+i) = 1e-5;
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
        Real Tpow = T > T_cond_max ? pow(T,2.5) : pow(T_cond_max,2.5);
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

