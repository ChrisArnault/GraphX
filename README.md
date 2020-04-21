# GraphX
Studies about Graph management, Pregel, Spark, Frames

Construct graph
===============

Grid
----
We split the space as a square grid of cells (currently: 100 x 100)

Vertices
--------
Vertices are created randomly, and a position (x, y) is assigned to them
Then we compute the cells

    conf.g = 100
    cell_id = lambda x, y: int(x*conf.g) + conf.g * int(y*conf.g)


We partition vertices against cell ids (300 partitions)

Edges
-----
**Edges are created first using an edge_iterator**:

* for all vertices:
  * select a random # of connected vertices from all vertices (up to ``max_degree``)
  * ignore self edges
* join [*source vertices, edges, dest vertices*]:
  * source vertex ``src.id == edge.src``
  * dest vertex ``dest.id == edge.dst``
  * source vertex ``src.id != dest.id``
  * source vertex cell is neighbour of dest vertex cell

**cell neighbours**:

two cells are said neighbours when:
* when they are adjacent by sides, ***or*** by corners
* including a continuous spheric space (left-right ***and*** top-down)

Batch management
----------------
Construction of large set of vertices and edges can be split in batches

* subset dataframes are created
* written (append mode) to hdfs(parquet)

Results
-------

1) Varying the number of vertices and the max-degree for edges (and batches for edges)

<table>
<thead>
<td>vertices</td>
<td>v_batches</td>
<td>V time</td>
<td>max_degree</td>
<td>e_batches</td>
<td>edges</td>
<td>total time</td>
<td>degree</td>
<td>triangles</td>
</thead>
<tr>
<td>1000</td>
<td>1</td>
<td>0h0m13.305s</td>
<td>100</td>
<td>1</td>
<td>14</td>
<td>0h0m10.283s</td>
<td>0h0m7.969s</td>
<td>0h0m5.313s</td>
</tr>
<tr>
<td>10000</td>
<td>10</td>
<td>0h0m54.576s</td>
<td>1000</td>
<td>10</td>
<td>1452</td>
<td>0h3m35.159s</td>
<td>0h0m7.735s</td>
<td>0h0m8.823s</td>
</tr>
<tr>
<td>100000</td>
<td>10</td>
<td>0h0m57.864s</td>
<td>1000</td>
<td>200</td>
<td>14749</td>
<td>0h42m32.747s</td>
<td>0h0m17.488s</td>
<td>0h0m31.310s</td>
</tr>
<tr>
<td>1000000</td>
<td>10</td>
<td>0h1m27.007s</td>
<td>1000</td>
<td>100</td>
<td>147045</td>
<td>4h33h24.873s</td>
<td>0h0m10.379s</td>
<td>0h0m47.097s</td>
</tr>
<tr>
<td>1000000</td>
<td>10</td>
<td>0h1m30.198s</td>
<td>1000</td>
<td>200</td>
<td>147003</td>
<td>4h47h24.070s</td>
<td>0h0m10.183s</td>
<td>0h0m26.816s</td>
</tr>
<tr>
<td>1000000</td>
<td>10</td>
<td>0h1m22.462s</td>
<td>10000</td>
<td>500</td>
<td>1470306</td>
<td>46h2h52.120s</td>
<td>0h0m19.660s</td>
<td>0h0m49.222s</td>
</tr>
</table>

2) Varying the number of edge-batches (same number of vertices [1000000] and degree[10000])

<table>
<thead>
<td>max_degree</td>
<td>e_batches</td>
<td>time per edge batch</td>
<td>total time</td>
</thead>
<tr>
<td>10000</td>
<td>500</td>
<td>5m</td>
<td>40h</td>
</tr>
<tr>
<td>10000</td>
<td>200</td>
<td>12m</td>
<td>42h</td>
</tr>
<tr>
<td>10000</td>
<td>100</td>
<td>27m</td>
<td>45h</td>
</tr>
<tr>
<td>20000</td>
<td>500</td>
<td>5m50</td>
<td>48h</td>
</tr>
</table>

GraphFrame
----------
Once vertices and edges are created, graphframes are assembled

Using graphs
============
A second application read vertex and edge dataframes and re-assemble graphframes and apply algorithms


