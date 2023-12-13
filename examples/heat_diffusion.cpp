#include <skepu>
#include <skepu-lib/io.hpp>
#include <mpi.h>

#define ENABLE_1D_EXAMPLE 1
#define ENABLE_2D_EXAMPLE 1
#define ENABLE_3D_EXAMPLE 1
#define ENABLE_4D_EXAMPLE 0

float heat1D(skepu::Region1D<float> r)
{
	float newval = 0;
	newval += r(-1);
	newval += r( 1);
	newval /= 2;
	return newval;
}

float heat2D(skepu::Region2D<float> r) {
	float newval = 0;
	newval += r(-1,  0);
	newval += r( 1,  0);
	newval += r( 0, -1);
	newval += r( 0,  1);
	newval /= 4;
	return newval;
}

float heat3D(skepu::Region3D<float> r) {
	float newval = 0;
	newval += r(-1,  0,  0);
	newval += r( 1,  0,  0);
	newval += r( 0, -1,  0);
	newval += r( 0,  1,  0);
	newval += r( 0,  0, -1);
	newval += r( 0,  0,  1);
	newval /= 6;
	return newval;
}

float heat4D(skepu::Region4D<float> r)
{
	float newval = 0;
	newval += r(-1,  0,  0,  0);
	newval += r( 1,  0,  0,  0);
	newval += r(0,  -1,  0,  0);
	newval += r(0,   1,  0,  0);
	newval += r(0,   0, -1,  0);
	newval += r(0,   0,  1,  0);
	newval += r(0,   0,  0, -1);
	newval += r(0,   0,  0,  1);
	newval /= 8;
	return newval;
}

int main(int argc, char *argv[])
{
    double timeStart = MPI_Wtime();

    MPI_Init(&argc, &argv); //initialize MPI operations

	if (argc < 5)
	{
		skepu::io::cout << "Usage: " << argv[0] << " dim size iterations backend\n";
		exit(1);
	}
	
	const int dim = atof(argv[1]);
	const int size = atof(argv[2]);
	const int iters = atof(argv[3]);
	const int output = atof(argv[5]);
	auto spec = skepu::BackendSpec{argv[4]};
        const int gpus = atof(argv[6]);
	
skepu::setGlobalBackendSpec(spec);

#if ENABLE_1D_EXAMPLE
	if (dim == 1)
	{
		auto update = skepu::MapOverlap(heat1D);
		update.setOverlap(1);
		update.setEdgeMode(skepu::Edge::None);
		update.setUpdateMode(skepu::UpdateMode::RedBlack);
		skepu::Vector<float> domain(size, 0);
		
		domain(0) = 0;
		domain(size-1) = 5;
		
		for (size_t i = 0; i < iters; ++i)
		{
			update(domain, domain);
		}
		
		skepu::io::cout << domain << "\n";
		exit(0);
	}
#endif
	
#if ENABLE_2D_EXAMPLE
	if (dim == 2)
	{

        auto update = skepu::MapOverlap(heat2D);
		update.setOverlap(1, 1);
        update.setEdgeMode(skepu::Edge::Pad);
        update.setPad(0);
        update.setUpdateMode(skepu::UpdateMode::Normal);
		skepu::Matrix<float> domain(size, size, 0);
        double beforefill = MPI_Wtime();
        double timeinit = beforefill-timeStart;

        for (size_t i = 0; i < size; ++i)
		{
			domain(i, 0) = 2;
			domain(i, size-1) = 2;
		}
		
		for (size_t i = 0; i < size; ++i)
		{
			domain(0, i) = 0;
			domain(size-1, i) = 5;
		}
        // Record stop event
        double init = MPI_Wtime();
        double timefill = init-beforefill;
		for (size_t i = 0; i < iters; ++i) {
            update(domain, domain);
            update(domain, domain);
		}
        double calc = MPI_Wtime();
        double timecalc = calc-init;

        // Write out result
        if (output) {
            skepu::external(skepu::read(domain), [&] {
                std::string fileName = "opencl-d2-s" + std::to_string(size) + "-i" + std::to_string(iters) + "-g" + std::to_string(gpus) + ".out";
                std::ofstream outputFile(fileName, std::ios::app); // append file or create a file if it does not exist
                for (int x = 0; x < size; x++) {
                    for (int y = 0; y < size; y++) {
                        float zelle = domain(x, y);
                        outputFile << zelle << ";"; // write
                    }
                    outputFile << "\n"; // write
                }
                outputFile.close();
            });
        }
        double endTime = MPI_Wtime();
        double totaltime = endTime - timeStart;

        std::string fileName = "runtime-opencl-d2-s" + std::to_string(size) + "-i" + std::to_string(iters) + "-g" + std::to_string(gpus) + ".out";
        std::ofstream outputFile(fileName, std::ios::app); // append file or create a file if it does not exist
        outputFile << "2" << ";" <<  size << ";" << iters << ";" << timeinit << ";" << timefill << ";" << timecalc << ";"
                   << totaltime << "\n"; // write
        outputFile.close();


        exit(0);
	}
#endif
	
#if ENABLE_3D_EXAMPLE
	if (dim == 3)
	{
        auto update = skepu::MapOverlap(heat3D);
		update.setOverlap(1, 1, 1);
        update.setEdgeMode(skepu::Edge::Pad);
        update.setPad(0);
        update.setUpdateMode(skepu::UpdateMode::Normal);
		skepu::Tensor3<float> domain(size, size, size, 0);
        double beforefill = MPI_Wtime();
        double timeinit = beforefill-timeStart;
		for (size_t i = 0; i < size; ++i) {
			for (size_t j = 0; j < size; ++j) {
				domain(0, i, j) = 1;
				domain(size-1, i, j) = 5;
				domain(i, 0, j) = 2;
				domain(i, size-1, j) = 2;
			}
		}
        double init = MPI_Wtime();
        double timefill = init-beforefill;

        for (size_t i = 0; i < iters; ++i) {
			update(domain, domain);
			update(domain, domain);
		}
        double calc = MPI_Wtime();
        double timecalc = calc-init;

        if (output) {
            skepu::external(skepu::read(domain), [&] {
                std::string fileName = "d3-opencl-s" + std::to_string(size) + "-i" + std::to_string(iters) + "-g" + std::to_string(gpus) + ".out";
                std::ofstream outputFile(fileName, std::ios::app); // append file or create a file if it does not exist
                for (int x = 0; x < size; x++) {
                    for (int y = 0; y < size; y++) {
                        for (int z = 0; z < size; z++) {
                            float zelle = domain(x, y, z);
                            outputFile << zelle << ";"; // write
                        }
                        outputFile << "\n"; // write
                    }
                }
                outputFile.close();
            });
        }
        double endTime = MPI_Wtime();
        double totaltime = endTime - timeStart;
        std::string fileName = "runtime-opencl-d3-s" + std::to_string(size) + "-i" + std::to_string(iters) + "-g" + std::to_string(gpus) + ".out";
        std::ofstream outputFile(fileName, std::ios::app); // append file or create a file if it does not exist
        outputFile << "3" << ";" <<  size << ";" << iters << ";" << timeinit << ";" << timecalc << ";"
                   << totaltime << "\n"; // write
        outputFile.close();
		exit(0);
	}
#endif

#if ENABLE_4D_EXAMPLE
	if (dim == 4)
	{
		auto update = skepu::MapOverlap(heat4D);
		update.setOverlap(1, 1, 1, 1);
		update.setEdgeMode(skepu::Edge::None);
		update.setUpdateMode(skepu::UpdateMode::RedBlack);
		skepu::Tensor4<float> domain(size, size, size, size, 0);
		
		for (size_t i = 0; i < size; ++i)
		{
			for (size_t j = 0; j < size; ++j)
			{
				for (size_t k = 0; k < size; ++k)
				{
					domain(0, i, j, k) = 1;
					domain(size-1, i, j, k) = 5;
					domain(i, 0, j, k) = 2;
					domain(i, size-1, j, k) = 2;
				}
			}
		}
		
		for (size_t i = 0; i < iters; ++i)
		{
			update(domain, domain);
		}
		
		//skepu::io::cout << domain << "\n";
		exit(0);
	}
#endif
	
	return 0;
}

