# Image Compression with Arithmetic Coding
## General Approach
To use arithmetic coding to encode an image. We built a state model of the pixel/quantized DCT values. Then, each 8x8 block in the image in encoded as a fractional value between \[0,1), representing the values in the block. Encoding using the pixel values creates a lossless compression, and using the quanitized DCT values creates a lossy one. Theoretically, the entire image could be encoded into a single fractional value, however, that creates a serious burden on the runtime.

## Implementation details
Storing rational numbers with the precision needed to encode the entire block is not possible with the limitations of a floating point. Instead, we calculated the upper and lower bounds of the encoded sequence using a fraction represented as a numerator and denominator. To find the shortest bit sequence that encodes a number between this point, we iteratively created a binary rational number, which essentially can be stored as just the numerator (since the denominator is a standard conversion).

## Results
For the lossless method, we compared the results to encoding each pixel value using exp-golomb of order 8 (this order produced the best results with exp-golomb on the test image).
For the lossy method, we compared the results to the run encoding DCT method.

## References:  
General Background:
- https://en.wikipedia.org/wiki/Arithmetic_coding
- http://home.ustc.edu.cn/~xuedixiu/image.ustc/course/dip/DIP14-ch7.pdf
- http://www.cs.ucf.edu/courses/cap5015/Arithmetic_coding_modified_2005.pdf

Basic implementation:
- https://www.researchgate.net/publication/200065260_Arithmetic_Coding_for_Data_Compression
- https://www.researchgate.net/publication/318643056_New_Image_CompressionDecompression_Technique_Using_Arithmetic_Coding_Algorithm

Implementation details:
- https://marknelson.us/posts/2014/10/19/data-compression-with-arithmetic-coding.html
- https://philipstel.wordpress.com/2010/12/21/arithmetic-coding-algorithm-and-implementation-issues-2/
- http://michael.dipperstein.com/arithmetic/index.html
