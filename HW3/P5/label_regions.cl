__kernel void
initialize_labels(__global __read_only int *image,
                  __global __write_only int *labels,
                  int w, int h)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if ((x < w) && (y < h)) {
        if (image[y * w + x] > 0) {
            // set each pixel > 0 to its linear index
            labels[y * w + x] = y * w + x;
        } else {
            // out of bounds, set to maximum
            labels[y * w + x] = w * h;
        }
    }
}

int get_clamped_value(__global __read_only int *labels,
                  int w, int h,
                  int x, int y)
{
    if ((x < 0) || (x >= w) || (y < 0) || (y >= h))
        return w * h;
    return labels[y * w + x];
}

// returns the smallest value of 5 values
int min_val(int a, int b, int c, int d, int e)
{
    int val1 = a;
    int val2 = d;
    int toret;
 
    if (b < a)
        val1 = b;
    if (e < d)
        val2 = e;
 
    if (val1 < val2)
        toret = val1;
    else
        toret = val2;
  
    if (c < toret)
        toret = c;
  
    return toret;
}

__kernel void
propagate_labels(__global __read_write int *labels,
                 __global __write_only int *changed_flag,
                 __local int *buffer,
                 int w, int h,
                 int buf_w, int buf_h,
                 const int halo)
{
    // halo is the additional number of cells in one direction

    // Global position of output pixel
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // coordinates of the upper left corner of the buffer in image
    // space, including halo
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // coordinates of our pixel in the local buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx;
    
    int old_label;
    // Will store the output value
    int new_label;
    
    // Load the relevant labels to a local buffer with a halo 
    if (idx_1D < buf_w) {
        for (int row = 0; row < buf_h; row++) {
            buffer[row * buf_w + idx_1D] = 
                get_clamped_value(labels,
                                  w, h,
                                  buf_corner_x + idx_1D, buf_corner_y + row);
        }
    }

    // Make sure all threads reach the next part after
    // the local buffer is loaded
    barrier(CLK_LOCAL_MEM_FENCE);

    int bg = w * h;
    int offset = buf_y * buf_w + buf_x;
    old_label = buffer[offset];
    
    // Fetch the value from the buffer the corresponds to
    // the pixel for this thread
    // CODE FOR PARTS 2
    //if (old_label < bg) {
    //    buffer[offset] = labels[buffer[offset]];
    //}   
    //barrier(CLK_LOCAL_MEM_FENCE);

    //CODE FOR PART 4
    int previdx = old_label; 
    int prevlabel = labels[old_label]; 
    
    if (idx_1D == 0) {
        for (int row = halo; row < buf_h - halo; row++) {
            for (int col = halo; col < buf_w - halo; col++) {
                int wg_off = row * buf_w + col;
                //check pixels in the foreground
                if (buffer[wg_off] < bg) {
                   //Fetches from grandparents when necessary
                   if(previdx != buffer[wg_off])
                       prevlabel = labels[buffer[wg_off]];
                   buffer[wg_off] = prevlabel;
                }
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    // to make readability easier for new_label
    int m = (buf_x + 0) + (buf_y + 0) * buf_w;
    int t = (buf_x + 0) + (buf_y - 1) * buf_w;
    int b = (buf_x + 0) + (buf_y + 1) * buf_w;
    int l = (buf_x - 1) + (buf_y + 0) * buf_w;
    int r = (buf_x + 1) + (buf_y + 0) * buf_w;

    // stay in bounds
    if ((x < w) && (y < h)) {
        // CODE FOR PART 1 HERE
        // We set new_label to the value of old_label, but you will need
        // to adjust this for correctness.
        new_label = old_label;
        if (new_label < bg) {
            new_label = min_val(buffer[m], buffer[t], buffer[b],
                                buffer[l], buffer[r]);
        }

        if (new_label != old_label) {
            // CODE FOR PART 3 HERE
            //update oldlabel to new label using atomic_min function
            atomic_min(old_label + labels, new_label);

            // indicate there was a change this iteration.
            // multiple threads might write this.
            *(changed_flag) += 1;
            labels[y * w + x] = new_label;
        }
    }
}
