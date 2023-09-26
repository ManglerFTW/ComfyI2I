self.onmessage = function(e) {
    const { imageData } = e.data;

    // Your image processing code here, for example:
    for (let i = 0; i < imageData.data.length; i += 4) {
        // Invert the alpha channel
        imageData.data[i+3] = 255 - imageData.data[i+3]; 
        
        // Setting the color channels to 0 to keep it a grayscale image
        imageData.data[i] = 0;
        imageData.data[i+1] = 0;
        imageData.data[i+2] = 0;
    }

    // Return the processed data back to the main thread
    self.postMessage({ processedData: imageData });
};