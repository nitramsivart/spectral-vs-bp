# Program to generate a community SBM in sparse form
# doesn't allow self edges
#
# Written by Mark Newman  2 JUL 2014
# Modified by Travis Martin  31 Jul 2014
# Modified by Travis Martin  8 Sep 2015

from numpy import ones,array
from random import randrange
from scipy.stats import poisson
from scipy.sparse import csr_matrix
from networkx import Graph

# n is total number of nodes
def fast_sbm(c_cc, c_cp, c_pp, n):
  mav_cc = c_cc * n / 4
  mav_cp = c_cp * n / 2
  mav_pp = c_pp * n / 4

  m_in1 = poisson.rvs(mav_cc)
  m_in2 = poisson.rvs(mav_pp)
  m_out = poisson.rvs(mav_cp)

  G = Graph()
  for i in range(n):
    G.add_node(i)

  # Generate first comm edges
  counter = 0
  while counter < m_in1:
    u = randrange(0,n//2)
    v = randrange(0,n//2)
    if u != v:
      G.add_edge(u,v)
      counter += 1

  # Generate second comm edges
  counter = 0
  while counter < m_in2:
    u = randrange(n//2,n)
    v = randrange(n//2,n)
    if u != v:
      G.add_edge(u,v)
      counter += 1

  # Generate between comm edges
  counter = 0
  while counter < m_out:
    u = randrange(0,n//2)
    v = randrange(n//2,n)
    if u != v:
      G.add_edge(u,v)
      counter += 1

  # Create sparse adjacency matrix
  return G
