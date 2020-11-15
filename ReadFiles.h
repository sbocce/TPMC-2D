
// This function will fill a vector with stuff read in a certain file

void FillMatrixFile(Eigen::MatrixXd& MAT, const char* filename)
{

// Matrices will be MAT(Ny, Nx), or in other words, MAT(R, z).

  size_t m, n;

  // Open file 
  std::ifstream f(filename);
  f >> m >> n; // Read dimensions as first elements, (Nx  Ny)

  // Resize matrix
  MAT.resize(m,n);
  
  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < n; j++)
    {
      f >> MAT(i,j);
    }
  }

  return;
}

// -----------------------------------------------------------

void FillVectorFile(Eigen::VectorXd& VEC, const char* filename)
{

  size_t m;

  // Open file 
  std::ifstream f(filename);
  f >> m; // First element is the dimension

  // Resize vector
  VEC.resize(m);
  
  for (int i = 0; i < m; i++)
  {
    f >> VEC(i);
  }

  return;
}


