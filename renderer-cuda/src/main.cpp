#include <cstdio>           
#include <fstream>          
#include <iostream>         
#include <sstream>          
#include <vector>           
#include <string>           
#include <unordered_map>    
#include <cmath>            
#include <cstring>          
#include <algorithm>        
#include <cuda_runtime.h>   
// Data structure to store physical properties of a single 3D Gaussian splat.
struct Gaussian {
    float x, y, z;          
    float scale[3];         
    float rot[4];           
    float r, g, b;          
    float opacity;          
};
extern "C" void launch_forward(
    int num_gaussians,           
    const float* d_positions,    
    const float* d_scales,       
    const float* d_rotations,    
    const float* d_colors,       
    const float* d_opacities,    
    const float* d_viewMatrix,   
    const float* d_projMatrix,   
    float* d_out_image,          
    int width,                   
    int height                   
);
#define CUDA_CHECK(val) check((val), #val, __FILE__, __LINE__)
// Helper function to intercept and print CUDA API errors.
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {    
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << err 
                  << " \"" << cudaGetErrorString(err) << "\" " << func << std::endl;
        std::exit(EXIT_FAILURE); 
    }
}
// Reads a .ply file, extracts Gaussian properties, and applies activation functions to convert them to physical values.
bool loadPLY(const std::string& filename, std::vector<Gaussian>& gaussians) {
    std::ifstream file(filename, std::ios::binary); 
    if (!file) {                                    
        std::cerr << "Failed to open PLY file: " << filename << "\n";
        return false;
    }
    std::string line;
    int num_vertices = 0;                           
    std::vector<std::pair<std::string, std::string>> properties; 
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') { 
            line.pop_back();                        
        }
        std::stringstream ss(line);                 
        std::string token;
        ss >> token;                                
        if (token == "element") {                   
            std::string element_name;
            ss >> element_name;
            if (element_name == "vertex") {
                ss >> num_vertices;                 
            }
        } else if (token == "property") {           
            std::string type, name;
            ss >> type >> name;
            properties.push_back({type, name});     
        } else if (token == "end_header") {         
            break;                                  
        }
    }
    if (num_vertices <= 0) {                        
        std::cerr << "Invalid vertex count in PLY header.\n";
        return false;
    }
    std::unordered_map<std::string, int> prop_offsets;
    int stride = 0; 
    for (const auto& prop : properties) {
        std::string type = prop.first;
        std::string name = prop.second;
        prop_offsets[name] = stride; 
        int size = 0;
        if (type == "float" || type == "float32" || type == "int" || type == "int32") size = 4;
        else if (type == "double") size = 8;
        else if (type == "uchar" || type == "uint8" || type == "char") size = 1;
        else if (type == "short" || type == "ushort") size = 2;
        else {
            std::cerr << "Unsupported property type: " << type << "\n";
            return false;
        }
        stride += size; 
    }
    auto check_and_get = [&](const std::string& name) -> int {
        auto it = prop_offsets.find(name);
        if (it == prop_offsets.end()) {
            std::cerr << "Required property not found in PLY: " << name << "\n";
            return -1;
        }
        return it->second;
    };
    int oX = check_and_get("x");
    int oY = check_and_get("y");
    int oZ = check_and_get("z");
    int oOpacity = check_and_get("opacity");
    int oS0 = check_and_get("scale_0");
    int oS1 = check_and_get("scale_1");
    int oS2 = check_and_get("scale_2");
    int oR0 = check_and_get("rot_0");
    int oR1 = check_and_get("rot_1");
    int oR2 = check_and_get("rot_2");
    int oR3 = check_and_get("rot_3");
    int oDc0 = check_and_get("f_dc_0"); 
    int oDc1 = check_and_get("f_dc_1");
    int oDc2 = check_and_get("f_dc_2");
    if (oX == -1 || oY == -1 || oZ == -1 || oOpacity == -1 ||
        oS0 == -1 || oS1 == -1 || oS2 == -1 ||
        oR0 == -1 || oR1 == -1 || oR2 == -1 || oR3 == -1 ||
        oDc0 == -1 || oDc1 == -1 || oDc2 == -1) {
        return false;
    }
    std::vector<char> record_buffer(stride); 
    gaussians.resize(num_vertices);          
    const float SH_C0 = 0.28209479177387814f;
    auto read_float = [&](const char* data, int offset) -> float {
        float val;
        std::memcpy(&val, data + offset, sizeof(float));
        return val;
    };
    auto clamp01 = [](float v) -> float {
        return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    };
    auto sigmoid = [](float v) -> float {
        return 1.0f / (1.0f + std::exp(-v));
    };
    for (int i = 0; i < num_vertices; ++i) {
        file.read(record_buffer.data(), stride);
        if (!file) {
            std::cerr << "Failed to read vertex " << i << " from PLY file.\n";
            return false;
        }
        const char* ptr = record_buffer.data();
        gaussians[i].x = read_float(ptr, oX);
        gaussians[i].y = read_float(ptr, oY);
        gaussians[i].z = read_float(ptr, oZ);
        gaussians[i].scale[0] = std::exp(read_float(ptr, oS0));
        gaussians[i].scale[1] = std::exp(read_float(ptr, oS1));
        gaussians[i].scale[2] = std::exp(read_float(ptr, oS2));
        gaussians[i].rot[0] = read_float(ptr, oR0);
        gaussians[i].rot[1] = read_float(ptr, oR1);
        gaussians[i].rot[2] = read_float(ptr, oR2);
        gaussians[i].rot[3] = read_float(ptr, oR3);
        float len = std::sqrt(gaussians[i].rot[0]*gaussians[i].rot[0] +
                              gaussians[i].rot[1]*gaussians[i].rot[1] +
                              gaussians[i].rot[2]*gaussians[i].rot[2] +
                              gaussians[i].rot[3]*gaussians[i].rot[3]);
        if (len > 0.f) {
            gaussians[i].rot[0] /= len;
            gaussians[i].rot[1] /= len;
            gaussians[i].rot[2] /= len;
            gaussians[i].rot[3] /= len;
        }
        gaussians[i].opacity = sigmoid(read_float(ptr, oOpacity));
        gaussians[i].r = clamp01(0.5f + SH_C0 * read_float(ptr, oDc0));
        gaussians[i].g = clamp01(0.5f + SH_C0 * read_float(ptr, oDc1));
        gaussians[i].b = clamp01(0.5f + SH_C0 * read_float(ptr, oDc2));
    }
    std::cout << "Successfully loaded " << num_vertices << " Gaussians from PLY file.\n";
    return true; 
}
// Writes the final 1D flattened RGB float array into a standard 2D .ppm image file.
void savePPM(const std::string& filename, const std::vector<float>& image, int width, int height) {
    std::ofstream out(filename, std::ios::binary); 
    if (!out) {
        std::cerr << "Failed to open output PPM file: " << filename << "\n";
        return;
    }
    out << "P6\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; ++i) {
        unsigned char r = static_cast<unsigned char>(std::max(0.0f, std::min(1.0f, image[3 * i + 0])) * 255.f);
        unsigned char g = static_cast<unsigned char>(std::max(0.0f, std::min(1.0f, image[3 * i + 1])) * 255.f);
        unsigned char b = static_cast<unsigned char>(std::max(0.0f, std::min(1.0f, image[3 * i + 2])) * 255.f);
        out.put(r);
        out.put(g);
        out.put(b);
    }
    std::cout << "Successfully saved rendered frame to " << filename << "\n";
}
// Main host execution: loads data, allocates GPU memory, copies data to device, launches rendering kernel, and saves the output.
int main() {
    std::string plyPath = "../../scenes/iceland_splat.ply";
    std::vector<Gaussian> gaussians; 
    if (!loadPLY(plyPath, gaussians)) {
        std::cerr << "Falling back to mock Gaussian data generation...\n";
        int mock_count = 500;
        gaussians.resize(mock_count);
        for (int i = 0; i < mock_count; ++i) {
            gaussians[i].x = ((float)rand() / RAND_MAX) * 4.f - 2.f;
            gaussians[i].y = ((float)rand() / RAND_MAX) * 4.f - 2.f;
            gaussians[i].z = ((float)rand() / RAND_MAX) * 5.f + 2.f; 
            gaussians[i].scale[0] = 0.05f;
            gaussians[i].scale[1] = 0.05f;
            gaussians[i].scale[2] = 0.05f;
            gaussians[i].rot[0] = 1.f; 
            gaussians[i].rot[1] = 0.f; 
            gaussians[i].rot[2] = 0.f; 
            gaussians[i].rot[3] = 0.f; 
            gaussians[i].r = (float)rand() / RAND_MAX;
            gaussians[i].g = (float)rand() / RAND_MAX;
            gaussians[i].b = (float)rand() / RAND_MAX;
            gaussians[i].opacity = 0.8f;
        }
    }
    int num_gaussians = static_cast<int>(gaussians.size());
    int width = 1280;  
    int height = 720; 
    std::vector<float> host_positions(num_gaussians * 3);
    std::vector<float> host_scales(num_gaussians * 3);
    std::vector<float> host_rotations(num_gaussians * 4);
    std::vector<float> host_colors(num_gaussians * 3);
    std::vector<float> host_opacities(num_gaussians);
    for (int i = 0; i < num_gaussians; ++i) {
        host_positions[3 * i + 0] = gaussians[i].x;
        host_positions[3 * i + 1] = gaussians[i].y;
        host_positions[3 * i + 2] = gaussians[i].z;
        host_scales[3 * i + 0] = gaussians[i].scale[0];
        host_scales[3 * i + 1] = gaussians[i].scale[1];
        host_scales[3 * i + 2] = gaussians[i].scale[2];
        host_rotations[4 * i + 0] = gaussians[i].rot[0];
        host_rotations[4 * i + 1] = gaussians[i].rot[1];
        host_rotations[4 * i + 2] = gaussians[i].rot[2];
        host_rotations[4 * i + 3] = gaussians[i].rot[3];
        host_colors[3 * i + 0] = gaussians[i].r;
        host_colors[3 * i + 1] = gaussians[i].g;
        host_colors[3 * i + 2] = gaussians[i].b;
        host_opacities[i] = gaussians[i].opacity;
    }
    float host_viewMatrix[16] = {
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 0.f,
        0.f, 0.f, 0.f, 1.f
    };
    float host_projMatrix[4] = {
        600.f, 600.f,               
        width / 2.f, height / 2.f   
    };
    float* d_positions = nullptr;
    float* d_scales = nullptr;
    float* d_rotations = nullptr;
    float* d_colors = nullptr;
    float* d_opacities = nullptr;
    float* d_viewMatrix = nullptr;
    float* d_projMatrix = nullptr;
    float* d_out_image = nullptr;
    CUDA_CHECK(cudaMalloc(&d_positions, num_gaussians * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scales, num_gaussians * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rotations, num_gaussians * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_colors, num_gaussians * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_opacities, num_gaussians * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_viewMatrix, 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_projMatrix, 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_image, width * height * 3 * sizeof(float))); 
    CUDA_CHECK(cudaMemcpy(d_positions, host_positions.data(), num_gaussians * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, host_scales.data(), num_gaussians * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rotations, host_rotations.data(), num_gaussians * 4 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colors, host_colors.data(), num_gaussians * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_opacities, host_opacities.data(), num_gaussians * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_viewMatrix, host_viewMatrix, 16 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_projMatrix, host_projMatrix, 4 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out_image, 0, width * height * 3 * sizeof(float)));
    std::cout << "Successfully allocated and copied " << num_gaussians << " Gaussians to GPU.\n";
    launch_forward(
        num_gaussians,
        d_positions, d_scales, d_rotations, d_colors, d_opacities,
        d_viewMatrix, d_projMatrix,
        d_out_image,
        width, height
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> host_image(width * height * 3); 
    CUDA_CHECK(cudaMemcpy(host_image.data(), d_out_image, width * height * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_positions));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_rotations));
    CUDA_CHECK(cudaFree(d_colors));
    CUDA_CHECK(cudaFree(d_opacities));
    CUDA_CHECK(cudaFree(d_viewMatrix));
    CUDA_CHECK(cudaFree(d_projMatrix));
    CUDA_CHECK(cudaFree(d_out_image));
    savePPM("rendered_output.ppm", host_image, width, height);
    std::printf("Host program execution complete.\n");
    return 0; 
}
