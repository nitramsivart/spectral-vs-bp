/// Computes the EM algorithm (using belief-propagation for the E step) for Kullback-Leibler divergence derived stochastic blockmodels.
// The base programming allows you to program in a mean-field approximation if you know how to. Unfortunately, the degree corrected stochastic blockmodel doesn't have an easy one that we can think of.
//  The standard stochastic blockmodel is programmed correctly, however.
//
//  currently just changing the basic blockmodel

#include <execinfo.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <list>
#include <limits>
#include "math.h"
#include <random>

using namespace std;

//double C_global = 1. / 5.;//n * c_inout / (n*(n-(1+c_inout)))
double C_global = 0;
double m_global = 0;

double ERROR = 1e-10;

struct Trio
{
  int out;
  int in;
  double known;
};

void check(double num, string str) {
  if(std::isnan(num)) {
    printf("found nan!!!!");
    cout << str << "\n";
    exit(0);
  }
  if(std::isinf(num)) {
    printf("found inf!!!!");
    cout << str << "\n";
    exit(0);
  }
  return;
  if(0 == num) {
    printf("found zero!!!!");
    cout << str << "\n";
    //exit(0);
  }
}
bool check_zero(double num, string str, double omega, double m1, double m2) {
  if(0 == num) {
    printf("found zero!!!!");
    //cout << str << " " << omega << " " << m1 << " " << m2 << "\n";
    //exit(0);
    return false;
  }
  return true;
}



double entropy(double x)
{
  if (x == 0)
  {
    return 0;
  }
  return x * log(x);
}


/* The M step has to reflect the function pointers as well! */
double SBM(const int adj_value, const double known, double& omega, double& k1, double& k2)
{
  if (adj_value == -1)
  {
    return 1;
  }
  // the below expression isn't defined for omega == 0, so we short circuit
  if (omega == 0) {
    //printf("short circuiting SBM\n");
    return 0;
  }
  return known * omega + C_global * (1. - known) *(1. - omega);
  //OLD: return exp(known * log(omega) + (1. - known) * log(1. - omega));
}

// This calculates the equivalent of omega, adjusted by degree
double DCSBM_P(double& omega, double& k1, double& k2) {
  double p = k1 * omega * k2 / (2.*m_global);
  if(p > 1) {
    //printf("%f %f\n",k1,k2);
    //printf("too large");
    return 1;
  }
  else
    return p;
}

// This is the Poisson formulation, which is why it looks different than the one above.
double DCSBM(const int adj_value, const double known, double& omega, double& k1, double& k2)
{
  if (adj_value == -1)
  {
    return 1;
  }
  // the below expression isn't defined for omega == 0, so we short circuit
  if (omega == 0)
    return 0;
  //cout << C_global << "  " << m_global << endl;
  double p = DCSBM_P(omega, k1, k2);
  return known * p + C_global * (1. - known) * (1. - p);
  //return exp(log(known * k1 * omega * k2 + C_global * (1-known)) - k1 * omega * k2) ;
}

// calculates q_rs^ij
double SBM_joint_marginal(double omega, double Q, double m1, double m2) {
  return (omega * Q + C_global * (1-omega) * (1-Q)) * m1 * m2;
}


/* M step initialization equations (can't assume the replica symmetric cavity equations since we don't have the messages) */
void M_SBM_init(Trio* EdgeList, double*** current_message, double** group_membership, double** omega, double* degrees, const int& vertices, const int& edges, const int& communities)
{
  double* nr = new double[communities];
  double** denom = new double*[communities];
  for (int i = 0; i < communities; i++)
  {
    nr[i] = 0;
    denom[i] = new double[communities];
    for (int j = 0; j < communities; j++)
    {
      denom[i][j] = 0;
      omega[i][j] = 0;
    }
  }

  for (int i = 0; i < vertices; i++)
  {
    for (int j = 0; j < communities; j++)
    {
      nr[j] += group_membership[i][j];
    }
  }

  for (int i = 0; i < edges; i++)
  {
    // Count towards m_rs
    if (EdgeList[i].known)
    {
      for (int j = 0; j < communities; j++)
      {
        for (int k = 0; k < communities; k++)
        {
          // Edges are undirected. I'm only buffering each edge once, so I need to count it twice, once for each direction.
          omega[j][k] += EdgeList[i].known * group_membership[EdgeList[i].out][j] * group_membership[EdgeList[i].in][k];
          omega[j][k] += EdgeList[i].known * group_membership[EdgeList[i].in][j] * group_membership[EdgeList[i].out][k];
        }
      }
    }
  }

  for (int j = 0; j < communities; j++)
  {
    for (int k = 0; k < communities; k++)
    {
      denom[j][k] += nr[j] * nr[k];
      if (denom[j][k] != 0)
        omega[j][k] /= denom[j][k];
      else
        printf("denom is 0, %d, %d\n", j, k);
    }
  }

  delete[] nr;
  for (int j = 0; j < communities; j++)
  {
    delete[] denom[j];
  }
  delete[] denom;
}

void M_DCSBM_init(Trio* EdgeList, double*** current_message, double** group_membership, double** omega, double* degrees, const int& vertices, const int& edges, const int& communities)
{
  double* kappar = new double[communities];
  double** denom = new double*[communities];
  for (int i = 0; i < communities; i++)
  {
    kappar[i] = 0;
    denom[i] = new double[communities];
    for (int j = 0; j < communities; j++)
    {
      denom[i][j] = 0;
      omega[i][j] = 0;
    }
  }

  for (int i = 0; i < vertices; i++)
  {
    for (int j = 0; j < communities; j++)
    {
      kappar[j] += degrees[i] * group_membership[i][j];
    }
  }

  for (int i = 0; i < edges; i++)
  {
    // Count towards m_rs
    if (EdgeList[i].known)
    {
      for (int j = 0; j < communities; j++)
      {
        for (int k = 0; k < communities; k++)
        {
          // Edges are undirected. I'm only buffering each edge once, so I need to count it twice, once for each direction.
          omega[j][k] += EdgeList[i].known * group_membership[EdgeList[i].out][j] * group_membership[EdgeList[i].in][k];
          omega[j][k] += EdgeList[i].known * group_membership[EdgeList[i].in][j] * group_membership[EdgeList[i].out][k];
        }
      }
    }
  }

  for (int j = 0; j < communities; j++)
  {
    for (int k = 0; k < communities; k++)
    {
      denom[j][k] += kappar[j] * kappar[k];
      if (denom[j][k] != 0)
        omega[j][k] /= (denom[j][k] / (2. * m_global));
    }
  }

  delete[] kappar;
  for (int j = 0; j < communities; j++)
  {
    delete[] denom[j];
  }
  delete[] denom;
}



/* M step equations */
void M_SBM(Trio* EdgeList, double*** current_message, double** group_membership, double** omega, double* degrees, const int& vertices, const int& edges, const int& communities)
{
  double* nr = new double[communities];
  double** denom = new double*[communities];
  double** omega_old = new double*[communities]; // this stores omega before M step
  double** omega_temp = new double*[communities]; // this stores omega within M step, from previous loop
  double denom_color;
  for (int i = 0; i < communities; i++)
  {
    omega_old[i] = new double[communities];
    omega_temp[i] = new double[communities];
    nr[i] = 0;
    denom[i] = new double[communities];
    for (int j = 0; j < communities; j++)
    {
      omega_old[i][j] = omega[i][j];
    }
  }

  for (int i = 0; i < vertices; i++)
  {
    for (int j = 0; j < communities; j++)
    {
      nr[j] += group_membership[i][j];
    }
  }

  for (int counter = 0; counter < 3; counter++) {
    for (int i = 0; i < communities; i++)
    {
      for (int j = 0; j < communities; j++)
      {
        denom[i][j] = 0;
        omega_temp[i][j] = omega[i][j];
        omega[i][j] = 0;
      }
    }
    //we set omega by another round of EM.
    for (int i = 0; i < edges; i++) {
      // non-edges shouldn't contribute to omega, and they give divide-by-zero errors
      if(EdgeList[i].known == 0) {
        continue;
      }
      denom_color = 0;
      for (int j = 0; j < communities; j++)
      {
        for (int k = 0; k < communities; k++)
        {
          denom_color += SBM_joint_marginal(omega_old[j][k], EdgeList[i].known, current_message[0][i][j], current_message[1][i][k]);
        }
      }
      for (int j = 0; j < communities; j++)
      {
        for (int k = 0; k < communities; k++)
        {
          double q0 = SBM_joint_marginal(omega_old[j][k], EdgeList[i].known, current_message[0][i][j], current_message[1][i][k]) / denom_color;
          double q1 = SBM_joint_marginal(omega_old[k][j], EdgeList[i].known, current_message[0][i][k], current_message[1][i][j]) / denom_color;


          // these are the joint marginals for edge / noneddge
          double qe = omega_temp[j][k] * EdgeList[i].known; 
          double qne = C_global * (1-omega_temp[j][k]) * (1-EdgeList[i].known);
          double norm = qe + qne;
          qe /= norm;
          qne /= norm;


          // Edges are undirected. I'm only buffering each edge once, so I need to count it twice, once for each direction.
          omega[j][k] += q0 * qe;
          omega[j][k] += q1 * qe;
          denom[j][k] += q0;
          denom[j][k] += q1;
        }
      }

    }

    for (int j = 0; j < communities; j++)
    {
      for (int k = 0; k < communities; k++)
      {
        denom[j][k] = nr[j] * nr[k];
        if (denom[j][k] != 0){
          omega[j][k] /= denom[j][k];
        }
        else {
          //printf("setting omega to 0\n");
          omega[j][k] = 0.;
        }
      }
    }

  }


  delete[] nr;
  for (int j = 0; j < communities; j++)
  {
    delete[] denom[j];
    delete[] omega_old[j];
    delete[] omega_temp[j];
  }
  delete[] denom;
  delete[] omega_old;
  delete[] omega_temp;
}



void M_DCSBM(Trio* EdgeList, double*** current_message, double** group_membership, double** omega, double* degrees, const int& vertices, const int& edges, const int& communities)
{
  double* kappar = new double[communities];
  double** denom = new double*[communities];
  double** omega_old = new double*[communities]; // stores omega before M step
  double** omega_temp = new double*[communities]; // stores omega within M step, from prev loop
  double denom_color;
  for (int i = 0; i < communities; i++)
  {
    omega_old[i] = new double[communities];
    omega_temp[i] = new double[communities];
    kappar[i] = 0;
    denom[i] = new double[communities];
    for (int j = 0; j < communities; j++)
    {
      omega_old[i][j] = omega[i][j];
    }
  }

  for (int i = 0; i < vertices; i++)
  {
    for (int j = 0; j < communities; j++)
    {
      kappar[j] += degrees[i] * group_membership[i][j];
    }
  }

  for (int counter = 0; counter < 1; counter++) {
    for (int i = 0; i < communities; i++)
    {
      for (int j = 0; j < communities; j++)
      {
        denom[i][j] = 0;
        omega_temp[i][j] = omega[i][j];
        omega[i][j] = 0;
      }
    }

    for (int i = 0; i < edges; i++)
    {
      // Count towards m_rs
      if (EdgeList[i].known != 0)
      {
        denom_color = 0;
        for (int j = 0; j < communities; j++)
        {
          for (int k = 0; k < communities; k++)
          {
            double p_old = DCSBM_P(omega_old[j][k], degrees[EdgeList[i].out], degrees[EdgeList[i].in]);
            denom_color += SBM_joint_marginal(p_old, EdgeList[i].known, current_message[0][i][j], current_message[1][i][k]);
          }
        }


        for (int j = 0; j < communities; j++)
        {
          for (int k = 0; k < communities; k++)
          {
            // this was omega for regular SBM. assume matrix is symmetric
            double p_old = DCSBM_P(omega_old[j][k], degrees[EdgeList[i].out], degrees[EdgeList[i].in]);

            // Edges are undirected. I'm only buffering each edge once, so I need to count it twice, once for each direction.
            double q0 = SBM_joint_marginal(p_old, EdgeList[i].known, current_message[0][i][j], current_message[1][i][k]) / denom_color;
            double q1 = SBM_joint_marginal(p_old, EdgeList[i].known, current_message[0][i][k], current_message[1][i][j]) / denom_color;


            double p_temp = DCSBM_P(omega_temp[j][k], degrees[EdgeList[i].out], degrees[EdgeList[i].in]);

            double qe = p_temp * EdgeList[i].known;
            double qne = C_global * (1-p_temp) * (1 - EdgeList[i].known);
            double norm = qe + qne;
            qe /= norm;

            omega[j][k] += q0*qe;
            omega[j][k] += q1*qe;
            denom[j][k] += q0;
            denom[j][k] += q1;
          }
        }
      }
    }

    for (int j = 0; j < communities; j++)
    {
      for (int k = 0; k < communities; k++)
      {
        denom[j][k] = kappar[j] * kappar[k];
        omega[j][k] *= 2. * m_global;
        if (denom[j][k] != 0)
          omega[j][k] /= denom[j][k];
        else{
          //printf("setting omega to 0\n");
          omega[j][k] = 0.;
        }
      }
    }
  }

  delete[] kappar;
  for (int j = 0; j < communities; j++)
  {
    delete[] denom[j];
    delete[] omega_old[j];
    delete[] omega_temp[j];
  }
  delete[] denom;
  delete[] omega_old;
  delete[] omega_temp;
}




/* Computing the mean field approximation terms */
void MF_SBM(double** degree_only, double** group_membership, double* degrees, double* missing_degrees, bool* degrees_present, double** omega, const int& vertices, const int& communities, double (*model)(const int, const double, double&, double&, double&))
{
  double part;
  double zero = 0.0;
  for (int j = 0; j < communities; j++)
  {
    degree_only[0][j] = 0;
  }
  for (int k = 0; k < communities; k++)
  {
    for (int i = 0; i < vertices; i++)
    {
      part = 0;
      for (int j = 0; j < communities; j++)
      {
        // Not degree-specific, so just throw it whatever for the degree.
        part += group_membership[i][j] * model(0, 0., omega[j][k], zero, degrees[i]);
      }
      if (part == 0)
      {
        //printf("ERROR1");
        part = 1;
      }

      degree_only[0][k] += log(part);
    }
  }
  return;
}




// Not actually a useful mean field algorithm. We have to do the full calculation since weighted degrees aren't integers
void MF_DCSBM(double** degree_only, double** group_membership, double* degrees, double* missing_degrees, bool* degrees_present, double** omega, const int& vertices, const int& communities, double (*model)(const int, const double, double&, double&, double&))
{
  double part;
  for (int i = 0; i < vertices; i++)
  {
    for (int k = 0; k < communities; k++)
    {
      degree_only[i][k] = 0;
    }

    for (int k = 0; k < communities; k++)
    {
      for (int j = 0; j < vertices; j++)
      {
        part = 0;
        for (int l = 0; l < communities; l++)
        {
          //check_zero(group_membership[j][l], "mf_group_membership",0,0,0);
          part += group_membership[j][l] * model(0, 0., omega[k][l], degrees[i], degrees[j]);
          //printf("gmemb, model: %f %f\n", group_membership[j][l], model(0,0,omega[k][l], degrees[i], degrees[j]));
        }
        if (part == 0) {
/*          if(group_membership[j][0] == 0 and group_membership[j][1] == 0 and group_membership[j][2] == 0)
            printf("G");*/
          //if(model(0,0,omega[k][0],degrees[i],degrees[j]) == 0) {
          //  printf("M %d %d %f %f", i, j, degrees[i], degrees[j]);
          //}
          //part = ERROR;
          part = 1;
          //printf("zero part\n");
        }
        degree_only[i][k] += log(part);
        //printf("i,part: %d %f\n", i, part);
        //check(degree_only[i][k], "degree_only");
      }
    }
  }
  return;
}

// If the network is unweighted, you can make use of the mean field approximation
void uMF_DCSBM(double** degree_only, double** group_membership, double* degrees, double* missing_degrees, bool* degrees_present, double** omega, const int& vertices, const int& communities, double (*model)(const int, const double, double&, double&, double&))
{
  double part;
  double thisdegree;
  for (int i = 0; i < vertices; i++)
  {
    thisdegree = i;
    if (degrees_present[i])
    {
      for (int k = 0; k < communities; k++)
      {
        degree_only[i][k] = 0;
      }

      for (int k = 0; k < communities; k++)
      {
        for (int j = 0; j < vertices; j++)
        {
          part = 0;
          for (int l = 0; l < communities; l++)
          {
            part += group_membership[j][l] * model(0, 0., omega[k][l], thisdegree, degrees[j]);
          }
          degree_only[i][k] += log(part);
        }
      }
    }
  }
  return;
}





/* Returning the mean field approximation terms */
double MFR_SBM(double** degree_only, const int& vertex, const int& community)
{
  return degree_only[0][community];
}


// Acts as the degree or the particular vertex, depending on whether the edges are weighted or not.
double MFR_DCSBM(double** degree_only, const int& vertex, const int& community)
{
  return degree_only[vertex][community];
}


// Computes the full log-likelihood
double LL_SBM(Trio* EdgeList, double** group_membership, double* nKcount, double** omega, double* degrees, const int& vertices, const int& edges, const int& communities)
{
  double likelihood = 0;
  // Should match nKcount since the function has converged, but it costs me nothing to calculate just in case.
  double* nx = new double[communities];
  for (int j = 0; j < communities; j++)
  {
    nx[j] = 0;
    for (int i = 0; i < vertices; i++)
    {
      nx[j] += group_membership[i][j];

      // Takes care of the entropy term.
      likelihood -= entropy(group_membership[i][j]);
    }

    // Takes care of the prior term: Q(Z)log(P(Z))
    if (nKcount[j])
    {
      likelihood += nx[j] * log(nKcount[j]);
    }
  }
  // Edges and the missing chunk from the previous term.
  for (int i = 0; i < edges; i++)
  {
    for (int j = 0; j < communities; j++)
    {
      for (int k = 0; k < communities; k++)
      {
        if (!(omega[j][k] == 0 || omega[j][k] == 1))
        {
          likelihood += group_membership[EdgeList[i].out][j] * group_membership[EdgeList[i].in][k] * log(EdgeList[i].known * omega[j][k] + C_global * (1. - EdgeList[i].known ) * (1. - omega[j][k]));
        }
      }
    }
  }

  delete[] nx;
  return likelihood;
}


double LL_DCSBM(Trio* EdgeList, double** group_membership, double* nKcount, double** omega, double* degrees, const int& vertices, const int& edges, const int& communities)
{
  double likelihood = 0;
  double* kappa = new double[communities];
  // Should match nKcount since the function has converged, but it costs me nothing to calculate just in case.
  double* nx = new double[communities];
  for (int j = 0; j < communities; j++)
  {
    kappa[j] = 0;
    nx[j] = 0;
    for (int i = 0; i < vertices; i++)
    {
      kappa[j] += group_membership[i][j] * degrees[i];
      nx[j] += group_membership[i][j];

      // Takes care of the entropy term.
      likelihood -= entropy(group_membership[i][j]);
    }
    printf("likelihood after entropy: %f\n", likelihood);

    // Takes care of the prior term: Q(Z)log(P(Z))
    if (nKcount[j])
    {
      likelihood += nx[j] * log(nKcount[j]);
    }
    printf("likelihood after prior: %f\n", likelihood);
  }

  printf("likelihood before model exp: %f\n", likelihood);

  // Term in the exponent of the model. Simplified to save time.
  for (int j = 0; j < communities; j++)
  {
    for (int k = 0; k < communities; k++)
    {
      likelihood -= kappa[j] * omega[j][k] * kappa[k] / (2*m_global);
    }
  }
  printf("likelihood before edges: %f\n", likelihood);

  // Edges and the missing chunk from the previous term.
  int print_once_count = 0;
  for (int i = 0; i < edges; i++)
  {
    for (int j = 0; j < communities; j++)
    {
      for (int k = 0; k < communities; k++)
      {
        double p = DCSBM_P(omega[j][k], degrees[EdgeList[i].out], degrees[EdgeList[i].in]);
          likelihood += group_membership[EdgeList[i].out][j] * group_membership[EdgeList[i].in][k] *
                                                               log(EdgeList[i].known * p + C_global * (1. - EdgeList[i].known) * (1. - p));
      }
    }
  }

  delete[] kappa;
  delete[] nx;
  return likelihood;
}



// Computes the number of vertices and edges in the network.
void FindStats(int& vertices, long int& edges, ifstream& file)
{
  char* buffer;
  long int entry1 = 0;
  long int entry2 = 0;
  string lineread;

  while(std::getline(file, lineread)) // Read line by line
  {
    buffer = new char [lineread.size()+1];
    strcpy(buffer, lineread.c_str());
    sscanf(buffer, "%ld %ld %*s", &entry1, &entry2); // "%*s" ignores the rest of the line.
    if (entry1 != entry2)
    {
      edges++;
    }
    if (entry1 > vertices)
    {
      vertices = entry1;
    }
    if (entry2 > vertices)
    {
      vertices = entry2;
    }
    delete[] buffer;
  }
  vertices++;
  file.clear();
  file.seekg(0, ios::beg);
  return;
}

// Sets the network.
void GetTheNetworkEdges(string fileName, int lines, Trio* EdgeList, double* degrees, double* missing_degrees, const int edges, const bool unweighted_degree)
{
  ifstream InputFile;
  string lineread;
  char *buffer;
  int count_edge = 0;
  long int entry1 = 0;
  long int entry2 = 0;
  float entry3 = 0;

  InputFile.open(fileName.c_str());
  if (!InputFile)
  {
    cout << "Error in opening file";
    cin.get();
    return;
  }

  // DON'T COUNT THE DEGREES FOR THE UNKNOWN EDGES!!! The estimation of the omega matrix uses the *OBSERVED* degree
  while(count_edge < edges && std::getline(InputFile, lineread)) // Read line by line
  {
    buffer = new char [lineread.size()+1];
    strcpy(buffer, lineread.c_str());
    // weighted case
    // put a ! here if we want this to be the mean field for the unweighted case
    // no ! means that we will be doing the mean field for the weighted case, by counting actual
    // 		weights even though we've told it not too
    if (lines > 2 && !unweighted_degree)
    {
      sscanf(buffer, "%ld %ld %f %*s", &entry1, &entry2, &entry3);
      if (entry1 != entry2)
      {
        EdgeList[count_edge].out = entry1;
        EdgeList[count_edge].in = entry2;
        EdgeList[count_edge].known = entry3;

        if (EdgeList[count_edge].known)
        {
          degrees[entry1] += entry3;
          degrees[entry2] += entry3;
        }
        else
        {
          missing_degrees[entry1]++;
          missing_degrees[entry2]++;
        }
      }
      else
      {
        count_edge--;
      }
    }
    else
    {
      sscanf(buffer, "%ld %ld %*s", &entry1, &entry2);
      if (entry1 != entry2)
      {
        EdgeList[count_edge].out = entry1;
        EdgeList[count_edge].in = entry2;
        EdgeList[count_edge].known = 1.;

        degrees[entry1]++;
        degrees[entry2]++;
      }
      else
      {
        count_edge--;
      }
    }

    delete[] buffer;
    count_edge++;
  }
  InputFile.close();

  return;
}


double Compute_Maxes(bool not_init, Trio* EdgeList, double*** current_message, double** group_membership, double* nKcount, double** omega, double* degrees, const int& vertices, const int& edges, const int& communities, void (*M_model)(Trio*, double***, double**, double**, double*, const int&, const int&, const int&))
{
  //// POTENTIAL FOR MEMORY LEAK
  double** omega_past = new double*[communities];
  double converged = 0;

  for (int j = 0; j < communities; j++)
  {
    omega_past[j] = new double[communities];
    for (int k = 0; k < communities; k++)
    {
      omega_past[j][k] = omega[j][k];
      if(std::isnan(omega_past[j][k]))
        printf("omega_past[%d][%d] is nan\n", j,k);
    }

    // should only use this for the simulated SBM
    //nKcount[j] = .5;
    nKcount[j] = 0;
    // the follwing is for real world stuff
    for (int i = 0; i < vertices; i++)
    {
      nKcount[j] += group_membership[i][j];
      //if(not_init)
        //printf("group memb: %f\n", group_membership[i][j]);
    }
    //printf("nKcount, vertices: %f, %d\n", nKcount[j], vertices);
    nKcount[j] /= double(vertices);
    //nKcount[j] = .5;
  }

  M_model(EdgeList, current_message, group_membership, omega, degrees, vertices, edges, communities);


  for (int j = 0; j < communities; j++)
  {
    for (int k = 0; k < communities; k++)
    {
      if (omega_past[j][k] >= 1e-10 && converged < fabs((omega[j][k] - omega_past[j][k]) / omega_past[j][k]))
      {
        converged = fabs((omega[j][k] - omega_past[j][k]) / omega_past[j][k]);
        if(std::isnan(converged) || std::isnan(omega[j][k]) || std::isnan(omega_past[j][k]))
          printf("found a nan\n");
      }
    }
  }

  //omega[0][0] = max(omega[0][0], omega[1][1]);
  //omega[1][1] = max(omega[0][0], omega[1][1]);

  //omega[0][0] = 0.04;
  //omega[1][1] = 0.04;
  //omega[0][1] = 0.028;
  //omega[1][0] = 0.028;

  for (int j = 0; j < communities; j++)
  {
    delete[] omega_past[j];
  }
  delete[] omega_past;

  return converged;

}







// For the messages, the first index (0) is for "out" sending to "in". (1) is for "in" sending to "out" (remember this is NOT symmetric!)
void BP_algorithm(Trio* EdgeList, double** group_membership, double** general_message, double*** current_message, double*** former_message, double* nKcount, double** omega, double** degree_only, double* degrees, double* missing_degrees, bool* degrees_present, double (*model)(const int, const double, double&, double&, double&), void (*MF_Precompute)(double**, double**, double*, double*, bool*, double**, const int&, const int&, double(const int, const double, double&, double&, double&)), double (*MF_Return)(double**, const int&, const int&), const int& vertices, const int& edges, const int& communities, const double& message_converged_diff, const double& zero_thresh, const bool& unweighted_degree)
{
  double max_diff = vertices;
  double partial_message_out, partial_message_in;
  double partial_message_out_denom, partial_message_in_denom;
  double norm1, norm2;
  double* temp_message1 = new double[communities];
  double* temp_message2 = new double[communities];
  bool set_zero;

  for (int i = 0; i < vertices; i++)
  {
    for (int j = 0; j < communities; j++)
    {
      general_message[i][j] = 0;
    }
  }

  int iteration = 0;
  while (max_diff > message_converged_diff && iteration < 20)
  {
    iteration++;
    // Pre-precompute the mean-field approximated term.
    MF_Precompute(degree_only, group_membership, degrees, missing_degrees, degrees_present, omega, vertices, communities, model);


    // Takes care of the mean-field term for the entire iteration!
    for (int i = 0; i < vertices; i++)
    {
      for (int j = 0; j < communities; j++)
      {
        if (unweighted_degree)
        {
          general_message[i][j] = MF_Return(degree_only, ceil(degrees[i]), j);
        }
        else
        {
          general_message[i][j] = MF_Return(degree_only, i, j);
          //printf("%d\n",i);
          //check(general_message[i][j], "gen_msg");
        }
      }
    }


    // precomupte the general messages.
    max_diff = 0;
    for (int i = 0; i < edges; i++)
    {
      if (EdgeList[i].known)
      {
        for (int j = 0; j < communities; j++)
        {
          partial_message_out = 0;
          partial_message_in = 0;
          partial_message_out_denom = 0;
          partial_message_in_denom = 0;
          for (int k = 0; k < communities; k++)
          {
            partial_message_out += former_message[1][i][k] * model(1, EdgeList[i].known, omega[j][k], degrees[EdgeList[i].out], degrees[EdgeList[i].in]);
            partial_message_in += former_message[0][i][k] * model(1, EdgeList[i].known, omega[j][k], degrees[EdgeList[i].in], degrees[EdgeList[i].out]);

            //check(partial_message_out, "partial_message_out");
            //check(partial_message_in, "partial_message_out");

            partial_message_out_denom += group_membership[EdgeList[i].in][k] * model(0, 0., omega[j][k], degrees[EdgeList[i].out], degrees[EdgeList[i].in]);
            partial_message_in_denom += group_membership[EdgeList[i].out][k] * model(0, 0., omega[j][k], degrees[EdgeList[i].in], degrees[EdgeList[i].out]);

            //check(partial_message_out, "partial_message_out_denom");
            //check(partial_message_in, "partial_message_out_denom");
          }
          // if the denominator is 0, then this means 'out' will never connect to any nodes if it's in community j
          // either because there are no nodes in k or omega[j][k] is 0. we set this to a default value,
          // need to make sure we set it to the same value in the mean field calculation
          if(partial_message_out == 0) {
            //printf("ERROR3.num");
            partial_message_out = 1;
          }
          if (partial_message_out_denom == 0)
          {
            //printf("ERROR3");
            partial_message_out_denom = 1;
          }
          general_message[EdgeList[i].out][j] += log(partial_message_out) - log(partial_message_out_denom);

          if (partial_message_in == 0) {
            //printf("ERROR4.num");
            partial_message_in = 1;
          }
          if (partial_message_in_denom == 0)
          {
            //printf("ERROR4");
            partial_message_in_denom = 1;
          }
          general_message[EdgeList[i].in][j] += log(partial_message_in) - log(partial_message_in_denom);
        }
      }
      else
      {
        for (int j = 0; j < communities; j++)
        {
          partial_message_out = 0;
          partial_message_in = 0;
          partial_message_out_denom = 0;
          partial_message_in_denom = 0;
          for (int k = 0; k < communities; k++)
          {
            partial_message_out_denom += group_membership[EdgeList[i].in][k] * model(0, 0., omega[j][k], degrees[EdgeList[i].out], degrees[EdgeList[i].in]);
            partial_message_in_denom += group_membership[EdgeList[i].out][k] * model(0, 0., omega[j][k], degrees[EdgeList[i].in], degrees[EdgeList[i].out]);
          }
          // is it possible for messages to be not known?
          general_message[EdgeList[i].out][j] -= log(partial_message_out_denom);
          general_message[EdgeList[i].in][j] -= log(partial_message_in_denom);
        }
      }
    }

    // Need to take off the self-edge contribution that we're not counting
    for (int i = 0; i < vertices; i++)
    {
      for (int j = 0; j < communities; j++)
      {
        partial_message_out_denom = 0;
        for (int k = 0; k < communities; k++)
        {
          partial_message_out_denom += group_membership[i][k] * model(0, 0., omega[j][k], degrees[i], degrees[i]);
        }
        if (partial_message_out_denom == 0)
        {//printf("ERROR5 %d %f\n",i, degrees[i]);
          partial_message_out_denom = 1;
          }

        general_message[i][j] -= log(partial_message_out_denom);
      }
    }


    // Then compute the actual messages
    for (int i = 0; i < edges; i++)
    {
      if (EdgeList[i].known)
      {
        for (int j = 0; j < communities; j++)
        {
          partial_message_out = 0;
          partial_message_in = 0;
          for (int k = 0; k < communities; k++)
          {
            partial_message_out += former_message[1][i][k] * model(1, EdgeList[i].known, omega[j][k], degrees[EdgeList[i].out], degrees[EdgeList[i].in]);
            partial_message_in += former_message[0][i][k] * model(1, EdgeList[i].known, omega[j][k], degrees[EdgeList[i].in], degrees[EdgeList[i].out]);
          }
          if(partial_message_out == 0)
            partial_message_out = 1;
          if(partial_message_in == 0)
            partial_message_in = 1;

          current_message[0][i][j] = general_message[EdgeList[i].out][j] - log(partial_message_out);
          current_message[1][i][j] = general_message[EdgeList[i].in][j] - log(partial_message_in);
        }

        for (int j = 0; j < communities; j++)
        {
          norm1 = nKcount[j];
          norm2 = nKcount[j];
          for (int k = 0; k < communities; k++)
          {
            if (j != k)
            {
              norm1 += nKcount[k] * exp(current_message[0][i][k] - current_message[0][i][j]);
              norm2 += nKcount[k] * exp(current_message[1][i][k] - current_message[1][i][j]);
            }
          }
          temp_message1[j] = nKcount[j] / norm1;
          temp_message2[j] = nKcount[j] / norm2;
          /*if(current_message[0][i][j] <= ERROR) {
            printf("no way");
            temp_message1[j] = .5;
          }
          else {
            temp_message1[j] = nKcount[j] / norm1;
          }

          if(current_message[0][i][j] <= ERROR) {
            printf("no way");
            temp_message2[j] = .5;
          }
          else {
            temp_message2[j] = nKcount[j] / norm2;
          }*/
        }

        for (int j = 0; j < communities; j++)
        {
          current_message[0][i][j] = temp_message1[j];
          current_message[1][i][j] = temp_message2[j];

          //check(current_message[0][i][j], "current_message[0]");
          //check(current_message[1][i][j], "current_message[1]");

          if (fabs(current_message[0][i][j] - former_message[0][i][j]) > max_diff)
          {
            max_diff = fabs(current_message[0][i][j] - former_message[0][i][j]);
          }

          if (fabs(current_message[1][i][j] - former_message[1][i][j]) > max_diff)
          {
            max_diff = fabs(current_message[1][i][j] - former_message[1][i][j]);
          }
        }
      }
    }


    for (int i = 0; i < edges; i++)
    {
      for (int j = 0; j < communities; j++)
      {
        former_message[0][i][j] = current_message[0][i][j];
        former_message[1][i][j] = current_message[1][i][j];
      }
    }


    for (int i = 0; i < vertices; i++)
    {
      for (int j = 0; j < communities; j++)
      {
        norm1 = nKcount[j];
        for (int k = 0; k < communities; k++)
        {
          if (k != j)
          {
            // we do the subtraction this way to avoid exponentiating large values
            // should work out to the same normalization factor
            norm1 += nKcount[k] * exp(general_message[i][k] - general_message[i][j]);
          }
        }
        temp_message1[j] = nKcount[j] / norm1;
      }

      set_zero = false;
      for (int j = 0; j < communities; j++)
      {
        group_membership[i][j] = temp_message1[j];
        if (group_membership[i][j] < zero_thresh)
        {
          group_membership[i][j] = 0;
          set_zero = true;
        }
        //check(group_membership[i][j], "group_membership");
      }

      if (set_zero == true)
      {
        norm2 = 0;
        for (int j = 0; j < communities; j++)
        {
          norm2 += group_membership[i][j];
        }

        for (int j = 0; j < communities; j++)
        {
          group_membership[i][j] /= norm2;
        }
      }
    }
  }


  delete[] temp_message1;
  delete[] temp_message2;

  return;
}







bool Sort_list_probability(Trio& a, Trio& b)
{
	if (a.known < b.known)
	{
		return false;
	}
	return true;
}

// current assumption: if original edge probability was 0, we aren't gonna predict an edge.
void predict_edges(Trio* EdgeList, double*** current_message, double** omega, double* degrees, const int& edges, const int& communities, double (*model)(const int, const double, double&, double&, double&), char * out_file_name) {
  ofstream out_file;
  out_file.open((string(out_file_name) + "_posterior.edges").c_str());

  for (int i = 0; i < edges; i++) {
    double edge_posterior = 0;

    if (EdgeList[i].known != 0) {
      double denom = 0; // a normalizing factor, for computing q_{ij}
      for (int j = 0; j < communities; ++j) {
        for (int k = 0; k < communities; ++k) {
          denom += SBM_joint_marginal(omega[j][k], EdgeList[i].known, current_message[0][i][j], current_message[1][i][k]);
        }
      }
      for (int j = 0; j < communities; ++j) {
        for (int k = 0; k < communities; ++k) {
          double p = omega[j][k];
          double q = SBM_joint_marginal(omega[j][k], EdgeList[i].known, current_message[0][i][j], current_message[1][i][k]) / denom;

          double t = p * EdgeList[i].known / (p * EdgeList[i].known + C_global * (1 - p) * (1 - EdgeList[i].known));
          edge_posterior += t * q;
        }
      }
    }
    out_file << EdgeList[i].out << " " << EdgeList[i].in << " " << edge_posterior << endl;
  }
  out_file.close();

}



int main(int argc, char* argv[])
{
  // Likelihood and M-step of the EM algorithm.
  // Only ever change this line to change the program's behavior. It consistently picks the model.
  // Right now this is only doing legacy stuff, but it's still needed to compile and I didn't want to change functionality too much.
  const bool degree_correction = false;
  // True if the edges are unweighted. This lets you use the mean field approximation for the DCSBM.
  const bool unweighted_degree = true;
  // True if we should run the edge prediction algorithm after running the normal community detection
  const bool predict_edges_flag = false;

  double (*model)(const int, const double, double&, double&, double&);
  void (*M_model_init)(Trio*, double***, double**, double**, double*, const int&, const int&, const int&);
  void (*M_model)(Trio*, double***, double**, double**, double*, const int&, const int&, const int&);
  void (*MF_Precompute)(double**, double**, double*, double*, bool*, double**, const int&, const int&, double(const int, const double, double&, double&, double&));
  double (*MF_Return)(double**, const int&, const int&);
  double (*Compute_Likelihood)(Trio*, double**, double*, double**, double*, const int&, const int&, const int&);

  // If you want the degree corrected version, add the changes here. I had been using the flag "degree_correction" above and an if-then statement here.
  if (degree_correction == false)
  {
    model = SBM;
    M_model_init = M_SBM_init;
    M_model = M_SBM;
    MF_Precompute = MF_SBM;
    MF_Return = MFR_SBM;
    Compute_Likelihood = LL_SBM;
  }
  else
  {
    model = DCSBM;
    M_model_init = M_DCSBM_init;
    M_model = M_DCSBM;
    if (unweighted_degree)
    {
      MF_Precompute = uMF_DCSBM;
    }
    else
    {
      MF_Precompute = MF_DCSBM;
    }
    MF_Return = MFR_DCSBM;
    Compute_Likelihood = LL_DCSBM;
  }





  int vertices = 0;
  long int edges = 0;
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_real_distribution<double> numgen(0.0,1.0);
  printf("rand: %f\n", numgen(generator));

  int communities = atoi(argv[2]);
  ifstream InputFile;
  InputFile.open(argv[1]);
  if (!InputFile)
  {
    cout << "Error in opening file";
    cin.get();
    return 0;
  }
  FindStats(vertices, edges, InputFile);
  InputFile.close();
  if (argc > 8)
  {
    vertices = atoi(argv[8]);
  }

  Trio* EdgeList = new Trio[edges];
  // This is the OBSERVED degree. The actual expected degree could be something different, depending on the number of unobserved vertex pairs.
  double* degrees = new double[vertices];
  double* missing_degrees = new double[vertices];
  // No vertex can have degree higher than the number of vertices
  bool* degrees_present = new bool[vertices];

  for (int i = 0; i < vertices; i++)
  {
    degrees[i] = 0;
    degrees_present[i] = false;
    missing_degrees[i] = 0;
  }

  GetTheNetworkEdges(string(argv[1]), atoi(argv[3]), EdgeList, degrees, missing_degrees, edges, unweighted_degree);

  // This is only useful if the degrees are already integers (i.e. the network is unweighted).
  for (int i = 0; i < vertices; i++)
  {
    // The addition of this ceil term is only to handle weighted 'histogram style' mean field approximating
    // I've removed it for the non-mean-field
    //degrees_present[int(ceil(degrees[i]))] = true;
    degrees_present[int(degrees[i])] = true;
  }

  // Whereas this used to be just the number of random starts on the E step, it's now random starts on the full EM algorithm
  int restarts = 10;
  if (argc > 4)
  {
    restarts = atoi(argv[4]);
  }


  // The program might cry and loop the EM algorithm infinitely unless you have a stopping condition somewhere.
  int max_iterations = 20;
  /*
  if (argc > 6)
  {
    max_iterations = atoi(argv[6]);
  }*/


  // For the convergence of the BP in the E step.
  double message_converged_diff = 0.01;
  /*
  if (argc > 7)
  {
    message_converged_diff = atof(argv[7]);
  }*/

  C_global = atof(argv[6]);
  m_global = atof(argv[7]);
  //n = atof(argv[6]);
  //C_global = c_inout / (n - 1. - c_inout);


  // For overall EM convergence.
  double converged_diff = 5e-2;
  if (argc > 8)
  {
    converged_diff = atof(argv[8]);
  }

  double zero_thresh = 1e-50;
  if (argc > 9)
  {
    zero_thresh = atof(argv[9]);
  }




  double** group_membership = new double*[vertices];
  double** best_groups = new double*[vertices];
  double** general_message = new double*[vertices];
  double** degree_only;


  if (degree_correction)
  {
    degree_only = new double*[vertices];
  }
  else
  {
    // Seems kind of pointless, but it makes the program more modular.
    degree_only = new double*[1];
  }
  for (int i = 0; i < vertices; i++)
  {
    group_membership[i] = new double[communities];
    best_groups[i] = new double[communities];
    general_message[i] = new double[communities];
    if (degree_correction)
    {
      degree_only[i] = new double[communities];
    }
  }
  if (!degree_correction)
  {
    degree_only[0] = new double[communities];
  }

  double* nKcount = new double[communities];
  double*** current_message = new double**[2];
  double*** former_message = new double**[2];
  double*** best_message = new double**[2];
  double norm1, norm2;

  current_message[0] = new double*[edges];
  current_message[1] = new double*[edges];
  former_message[0] = new double*[edges];
  former_message[1] = new double*[edges];
  best_message[0] = new double*[edges];
  best_message[1] = new double*[edges];
  for (int i = 0; i < edges; i++)
  {
    current_message[0][i] = new double[communities];
    current_message[1][i] = new double[communities];
    former_message[0][i] = new double[communities];
    former_message[1][i] = new double[communities];
    best_message[0][i] = new double[communities];
    best_message[1][i] = new double[communities];
  }

  double** omega = new double*[communities];
  double** best_omega = new double*[communities];
  for (int j = 0; j < communities; j++)
  {
    omega[j] = new double[communities];
    best_omega[j] = new double[communities];
  }


  double LL, best_LL;
  int EMiterations;

  for (int thisiteration = 0; thisiteration < restarts; thisiteration++)
  {	
    // Begin (randomized) initialization steps
    printf("iteration %d\n", thisiteration);
    for (int j = 0; j < communities; j++)
    {
      nKcount[j] = 0;
    }

    
    for (int i = 0; i < vertices; i++)
    {
      norm1 = 0;
      for (int j = 0; j < communities; j++)
      {
        group_membership[i][j] = 1; // numgen(generator);
        norm1 += group_membership[i][j];
      }
      for (int j = 0; j < communities; j++)
      {
        group_membership[i][j] /= norm1;
        nKcount[j] += group_membership[i][j];
      }
    }

    for (int j = 0; j < communities; j++)
    {
      nKcount[j] /= double(vertices);
    }

    /*
    for (int j = 0; j < communities; j++)
    {
      nKcount[j] = .5;// /= double(vertices);
    }*/

    for (int j = 0; j < edges; j++)
    {
      norm1 = 0;
      norm2 = 0;
      for (int k = 0; k < communities; k++)
      {
        former_message[0][j][k] = numgen(generator);
        former_message[1][j][k] = numgen(generator);

        //current message needs to be reset each time
        current_message[0][j][k] = 0;
        current_message[1][j][k] = 0;

        norm1 += former_message[0][j][k];
        norm2 += former_message[1][j][k];
      }
      for (int k = 0; k < communities; k++)
      {
        former_message[0][j][k] /= norm1;
        former_message[1][j][k] /= norm2;
      }
    }

    // why is omega set to 0 each time?? this gives problems
    // in the first run of compute maxes, which here is essentially
    // just skipping things
    for (int j = 0; j < communities; j++)
    {
      for (int k = 0; k < communities; k++)
      {
        omega[j][k] = 0;
      }
    }

    // Start with the M step. Except this should be the non-BP version, whereas the full EM algorithm one should include the messages as expected.
    double converged = Compute_Maxes(false, EdgeList, current_message, group_membership, nKcount, omega, degrees, vertices, edges, communities, M_model_init);
    printf("Init omega!\n");
    for(int i=0; i < communities; i++) {
      for(int j=0; j < communities; j++) {
        cout << omega[i][j] << "\t";
      }
      cout << "\n";
    }
    cout << "gamma\n";

    for(int i = 0; i < communities; i++) {
      cout << nKcount[i] << "\t";
    }
    cout << endl;

    converged = converged_diff * 4;
    for (int i = 0; i < communities; i++)
    {
      for (int j = 0; j < communities; j++)
      {
        if (i != j)
        {
          omega[i][j] *= 0.5;
        }
      }
    }
    EMiterations = 0;
    if (edges != 0)
    {
      while (converged > converged_diff && EMiterations < max_iterations)
      {
        printf("converged %f, EMiteration %d\n", converged, EMiterations);
        BP_algorithm(EdgeList, group_membership, general_message, current_message, former_message, nKcount, omega, degree_only, degrees, missing_degrees, degrees_present, model, MF_Precompute, MF_Return, vertices, edges, communities, message_converged_diff, zero_thresh, unweighted_degree);
        converged = Compute_Maxes(true, EdgeList, current_message, group_membership, nKcount, omega, degrees, vertices, edges, communities, M_model);
        for(int i=0; i < communities; i++) {
          for(int j=0; j < communities; j++) {
            cout << omega[i][j] << "\t";
          }
          cout << "\n";
        }
        cout << "gamma\n";

        for(int i = 0; i < communities; i++) {
          cout << nKcount[i] << "\t";
        }
        cout << endl;

        EMiterations++;
      }

      LL = Compute_Likelihood(EdgeList, group_membership, nKcount, omega, degrees, vertices, edges, communities);
    }
    else
    {
      for (int i = 0; i < vertices; i++)
      {
        for (int j = 0; j < communities; j++)
        {
          group_membership[i][j] = 1. / communities;
        }
      }

      LL = Compute_Likelihood(EdgeList, group_membership, nKcount, omega, degrees, vertices, edges, communities);
    }

    if (std::isnan(LL))
    {
      LL = -std::numeric_limits<float>::max();
    }

    if (thisiteration == 0)
    {
      best_LL = LL;
      for (int j = 0; j < communities; j++)
      {
        for (int i = 0; i < vertices; i++)
        {
          best_groups[i][j] = group_membership[i][j];
        }

        for (int k = 0; k < communities; k++)
        {
          best_omega[j][k] = omega[j][k];
        }
        for (int i = 0; i < edges; i++)
        {
          best_message[0][i][j] = current_message[0][i][j];
          best_message[1][i][j] = current_message[1][i][j];
        }
      }
    }
    if (thisiteration == 0 || LL > best_LL)
    {
      best_LL = LL;
      for (int j = 0; j < communities; j++)
      {
        for (int i = 0; i < vertices; i++)
        {
          best_groups[i][j] = group_membership[i][j];
        }

        for (int k = 0; k < communities; k++)
        {
          best_omega[j][k] = omega[j][k];
        }
        for (int i = 0; i < edges; i++)
        {
          best_message[0][i][j] = current_message[0][i][j];
          best_message[1][i][j] = current_message[1][i][j];
        }
      }
      ofstream out_file;
      if (argc > 5)
      {
        out_file.open(argv[5]);
      }
      if (out_file.is_open())
      {
        for (int i = 0; i < vertices; i++)
        {
          for (int j = 0; j < communities; j++)
          {
            if (best_groups[i][j] < zero_thresh)
            {
              out_file << "0  ";
            }
            else
            {
              out_file << best_groups[i][j] << "  ";
            }
          }
          out_file << endl;
        }
      }
      out_file.close();

      if(predict_edges_flag == true)
        predict_edges(EdgeList, best_message, best_omega, degrees, edges, communities, model, argv[5]);
    }
    printf("LL: %f, Best LL: %f\n", LL, best_LL);
  }



  for (int i = 0; i < communities; i++)
  {
    delete[] omega[i];
    delete[] best_omega[i];
  }
  delete[] omega;
  delete[] best_omega;
  for (int i = 0; i < vertices; i++)
  {
    delete[] group_membership[i];
    delete[] best_groups[i];
    delete[] general_message[i];
  }
  for (int i = 0; i < edges; i++)
  {
    delete[] current_message[0][i];
    delete[] current_message[1][i];
    delete[] former_message[0][i];
    delete[] former_message[1][i];
    delete[] best_message[0][i];
    delete[] best_message[1][i];
  }
  delete[] group_membership;
  delete[] best_groups;
  delete[] general_message;
  delete[] current_message;
  delete[] former_message;
  delete[] best_message;
  delete[] degrees;
  delete[] degrees_present;
  delete[] missing_degrees;
  if (degree_correction)
  {
    for (int i = 0; i < vertices; i++)
    {
      delete[] degree_only[i];
    }
  }
  else
  {
    delete[] degree_only[0];
  }
  delete[] degree_only;
  delete[] EdgeList;
  return 0;
}
