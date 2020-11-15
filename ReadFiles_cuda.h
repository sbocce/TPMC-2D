// This function will fill a vector with stuff read in a certain file

void FillMatrixFile(fpTYPE* h_MAT, const char* filename)
{

// Matrices will be a vector of Nx*Ny elements. 

  size_t m, n;

  // Open file 
  std::ifstream f(filename);
  f >> m >> n; // Read dimensions as first elements, (Nx  Ny)

  // Read one element after the other and fill the matrix
  for (int ID = 0; ID < m*n; ++ID)
  {
    f >> h_MAT[ID];
  }
 
  return;
}
 
// -----------------------------------------------------------

void FillVectorFile(fpTYPE* h_VEC, const char* filename)
{
  size_t m;

  // Open file 
  std::ifstream f(filename);
  f >> m; // First line is the vector dimension

  for (int i = 0; i < m; i++)
  {
    f >> h_VEC[i];
  }

  return;
}

