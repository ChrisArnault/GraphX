# GraphX
Studies about Graph management, Pregel, Spark, Frames

Construct graph
===============

Grid
----
We split the space as a square grid of cells

Vertices
--------
Vertices are created randomly, and a position (x, y) is assigned to them
Then we compute the cells

We partition vertices against cell ids (300 partitions)

Edges
-----
Edges are createdfirst using an edge_iterator:

* for all vertices:
  * select a random # of connected vertices from all vertices (up to a max_degree)
  * ignore self edges
* join source vertices, edges, dest vertices:
  * source vertex id == edge.src, dest vertex id == edge.dst
  * source vertex id != dest vertex id == edge.dst
  * source vertex cell is neighbour of dest vertex cell

cell neighbours:

two cells are said neighbours
* when they are adjacent by sides, or by corners
* including spheric space (left-right and top-down)

Batch management
----------------
Construction of large vertices, edges can be split in batches

* subset dataframes are created
* written (append mode) to hdfs(parquet)

Results
-------
<table>
<thead>
<td>vertices</td>
<td>v_batches</td>
<td>V time</td>
<td>max_degree</td>
<td>e_batches</td>
<td>edges</td>
<td>total time</td>
</thead>
<tr>
<td>1000</td>
<td>1</td>
<td>0h0m13.305s</td>
<td>100</td>
<td>1</td>
<td>14</td>
<td>0h0m10.283s</td>
</tr>
<tr>
<td>10000</td>
<td>10</td>
<td>0h0m48.059s</td>
<td>1000</td>
<td>10</td>
<td>1444</td>
<td>0h3m27.544s</td>
</tr>
<tr>
<td>100000</td>
<td>10</td>
<td></td>
<td>1000</td>
<td>100</td>
<td></td>
<td></td>
</tr>
<tr>
<td>1000000</td>
<td>10</td>
<td>0h1m27.007s</td>
<td>1000</td>
<td>100</td>
<td>147045</td>
<td>4h33h24.873s</td>
</tr>
</table>

GraphFrame
----------
Once vertices and edges are created, graphframes are assembled

Using graphs
============
A second application read vertex and edge dataframes and re-assemble graphframes and apply algorithms


