#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_BaseFab.H>
#include <AMReX_GpuControl.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>

//using namespace amrex;

void launchKernels(amrex::MultiFab* mf, amrex::Vector<int>* runOnGpu)
{
    BL_PROFILE_VAR("launchKernels", blp);

    //amrex::Gpu::LaunchSafeGuard lsg(false);
    
    for (amrex::MFIter mfi(*mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Gpu::FuseSafeGuard fsg(false);
        
        const amrex::Box& bx = mfi.tilebox();
        
        amrex::Array4<amrex::Real> const& fab = (*mf).array(mfi);

        using namespace amrex;
        
        amrex::RunOn run_on;
        // if ((*runOnGpu)[mfi.index()] == 1) {
        //     amrex::Print() << "the device!\n";
        //     run_on = amrex::RunOn::Device;
        // }
        // else {
        //     amrex::Print() << "the host!\n";
        //     run_on = amrex::RunOn::Host;
        // }
        run_on = ((*runOnGpu)[mfi.index()] == 1)? amrex::RunOn::Device : amrex::RunOn::Host;
            

        //AMREX_HOST_DEVICE_PARALLEL_FOR_4D(bx, mf->nComp(), i, j, k, n,
        AMREX_HOST_DEVICE_PARALLEL_FOR_4D_FLAG(run_on, bx, mf->nComp(), i, j, k, n,
        {
            fab(i,j,k,n) += 1.;
            //d(i,j,k,n+destcomp) = s(i+offset.x,j+offset.y,k+offset.z,n+srccomp);
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
    amrex::Vector<int> runOnGpu(ba.size(), 0);


    // Todo:
    // Measure costs on GPU and CPU
    // Load balance according to 'multidimensional dm'
    // Run with computed distribution mapping
    // compared to CPU only, GPU only
    // make sure instrumenting the runtime appropriately
    
    launchKernels(&mf, &runOnGpu);
    
    //cudaDeviceSynchronize();
    
    amrex::Finalize();
}

