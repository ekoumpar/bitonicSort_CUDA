# Bitonic Sort implemented in CUDA

## Project Description

This project implements a parallel sorting program using **CUDA**, based on the Bitonic Sort algorithm. The program sorts $N = 2^q$ integers in ascending order, leveraging GPU parallelism to accelerate the sorting process. Three versions of the algorithm (V0, V1, V2) are developed, each introducing optimizations to improve efficiency. 

The implementation combines thread-level parallelism, shared memory usage, and kernel synchronization to achieve high-performance sorting. The program validates correctness and compares execution times for different input sizes ($q \in [20, 27]$) against the quickSort (qSort) algorithm. A summary of the results can be seen below. 

For a detailed analysis of the algorithm and results, refer to the project [report](documentation/report.pdf)

---

## Instructions
The following instructions should be executed in the parent folder of the repository, where Makefile is found.

1. **Compile the program** using the following command. Subsitute <VERSION> with V0, V1, V2 or qSort:
    ```bash
    make <VERSION>
    ```

2. **Run the program** using the following command. Subsitute <q> with a positive integer greater than 10:
    ```bash
    make run q=<q>
    ```

3. **Expected Output** Example for VERSION=V0 and q=20:
   ```bash
    Running V0 with q=20...
    Sorted array
    Execution time: 0.013430 s
   ```

4. **Clean up** the directory of the files generated using the following command:
    ```bash
    make clean
    ```

   **Note**: Running  ```make``` will also display these instruction along with an example.

---

## Results Summary

This project was tested using **Google Colab** and the and **Aristotle Cluster**. Google Colab is using **NVIDIA Tesla T4 GPU** and Aristotle Cluster is using **NVIDIA Tesla P100 GPU** with the following specs:

 GPU Type           | NVIDIA Tesla T4    |   NVIDIA Tesla P100     |
 -------------------|--------------------|-------------------------|
 Cores              |    2560            |    3584                 |
 RAM                |    16GB            |    16GB                 |    
 CUDA Version       |    7.5             |    6.0                  |
 
 
 Below is a summary of the testing results:

###  V0 

q ($2^q$ elements) |     20     | 	21       | 	22      |    23	       |   24      |  25       |  26       |  27     |
-------------------|------------|------------|----------|--------------|-----------|-----------|-----------|---------| 
Execution time (s) |
Google Collab      | 0.013      |	0.028      | 0.057    |  0.122       |  0.257    |  0.433    |  0.823  | 1.677   |
Aristotel Cluster  | 0.013      |	0.019    | 0.036    |  0.067       |  0.133    |  0.267    |  0.552    | 1.269   |


###  V1

q ($2^q$ elements) |     20     | 	21       | 	22      |    23	       |   24      |  25       |  26       |  27     |
-------------------|------------|------------|----------|--------------|-----------|-----------|-----------|---------| 
Execution time (s) |
Google Collab      |  0.009      |	0.019    | 0.041    |   0.089      |  0.186    |   0.357   |   0.629  |  1.234   |
Aristotel Cluster  |  0.013      |	0.018    | 0.030    |   0.061      |  0.120    |  0.236    |  0.506   |  1.047   |  

###  V2

q ($2^q$ elements) |     20     | 	21       | 	22      |    23	       |   24      |  25       |  26       |  27     |
-------------------|------------|------------|----------|--------------|-----------|-----------|-----------|---------| 
Execution time (s) |
Google Collab      |  0.011     |	0.024    | 0.052    |   0.112      |  0.238    |   0.418   |  0.708    |  1.390  |
Aristotel Cluster  |  0.012     |	0.019    | 0.034    |  0.067       |  0.135    |  0.262    |  0.571    |  1.170  |     

For a detailed analysis please refer to the project report.

---

## Credits 

Written by Koumparidou Eleni and Kostomanolaki Maria Sotiria in January 2025.
