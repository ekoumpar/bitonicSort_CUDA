# bitonicSort_CUDA

## Makefile Instructions

1. **Compile the program** using the following command. Subsitute <VERSION> with V0, V1 or V2:
    ```bash
    make <VERSION>
    ```

2. **Run the program** using the following command. Subsitute <q> with a positive integer greater than 10:
    ```bash
    make run q=<q>
    ```

3. **Expected Output** Example for V=0 and q=20:
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

## Google Colab testing 
###  V0 - 2^q ELEMENTS

q VALUE            |     20     | 	21       | 	22      |    23	       |   24      |  25       |  26       |  27     |
-------------------|------------|------------|----------|--------------|-----------|-----------|-----------|---------| 
EXECUTION TIME (s) | 0.013      |	0.028      | 0.057    |  0.122       |  0.257    |  0.433    |  0.823    | 1.677   |

## Google Colab testing 
### V1 - 2^q ELEMENTS

q VALUE            |     20     | 	21       | 	22      |    23	       |   24      |  25       |  26       |  27     |
-------------------|------------|------------|----------|--------------|-----------|-----------|-----------|---------| 
EXECUTION TIME (s) | 0.010      |	0.021      | 0.045    |  0.096       |  0.208    |  0.358    |  0.650    | 1.262   |
