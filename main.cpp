#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuControl.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>


void launchKernels(amrex::MultiFab* mf, amrex::Vector<int>* runOnGpu)
{
    BL_PROFILE_VAR("launchKernels", blp);
    
    // Todo:
    // Load balance according to list of [[cost_gpu_1, cost_cpu_1], [cost_gpu_2, cost_cpu_2], ... ]
    // Run with computed distribution mapping
    // Need to control whether executing on CPU or GPU
    
    // Seems safeguard only makes a differene called from outside
    amrex::Gpu::LaunchSafeGuard lsg(false);
    for (amrex::MFIter mfi(*mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {   
        // Then call right function RunOn::Device or RunOn::Host
        const amrex::Box& bx = mfi.tilebox();
        
        amrex::Array4<amrex::Real> const& fab = (*mf).array(mfi);
        
        amrex::ParallelFor(bx, mf->nComp(),
                           [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
                           {
                               fab(i,j,k,n) += 1.;
                           });        
    }
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    // Initialize BoxArrays from data
    //std::ifstream ifs("BoxData/ba.60", std::ios::in);
    // std::ifstream ifs("BoxData/ba.213", std::ios::in);
    // std::ifstream ifs("BoxData/ba.1000", std::ios::in);
    // std::ifstream ifs("BoxData/ba.3865", std::ios::in);
    //std::ifstream ifs("BoxData/ba.5034", std::ios::in);
    std::ifstream ifs("BoxData/ba.15456", std::ios::in);
    // std::ifstream ifs("BoxData/ba.mac.294", std::ios::in);
    // std::ifstream ifs("BoxData/ba.23925", std::ios::in);

    amrex::BoxArray ba;
    ba.readFrom(ifs);

    // Distribution mapping: KNAPSACK, SFC
    amrex::DistributionMapping::strategy(amrex::DistributionMapping::KNAPSACK);
    amrex::DistributionMapping dm(ba);

    // Number of components, number of ghost cells
    int ncomp = 3;
    int ngrow = 1;

    // Initialize the MultiFab
    amrex::MultiFab mf(ba, dm, ncomp, ngrow);

    std::cout << "# of grids: " << ba.size() << '\n';

    // Set the runOnGpu decisions
    amrex::Vector<int> runOnGpu(ba.size(), 1);
    
    launchKernels(&mf, &runOnGpu);
    
    cudaDeviceSynchronize();
    
    amrex::Finalize();
}

