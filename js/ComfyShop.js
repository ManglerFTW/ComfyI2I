import { app } from "../../scripts/app.js";
import { ComfyDialog, $el } from "../../scripts/ui.js";
import { ComfyApp } from "../../scripts/app.js";
import { api } from "../../scripts/api.js"
import { ClipspaceDialog } from "../../extensions/core/clipspace.js";

////
function addMenuHandler(nodeType, cb) {
	const getOpts = nodeType.prototype.getExtraMenuOptions;
	nodeType.prototype.getExtraMenuOptions = function () {
		const r = getOpts.apply(this, arguments);
		cb.apply(this, arguments);
		return r;
	};
}

async function uploadMask(filepath, formData) {
	await api.fetchApi('/upload/mask', {
		method: 'POST',
		body: formData
	}).then(response => {}).catch(error => {
		console.error('Error:', error);
	});

	ComfyApp.clipspace.imgs[ComfyApp.clipspace['selectedIndex']] = new Image();
	ComfyApp.clipspace.imgs[ComfyApp.clipspace['selectedIndex']].src = api.apiURL("/view?" + new URLSearchParams(filepath).toString() + app.getPreviewFormatParam());
	if(ComfyApp.clipspace.images)
		ComfyApp.clipspace.images[ComfyApp.clipspace['selectedIndex']] = filepath;

	ClipspaceDialog.invalidatePreview();
}

// Helper function to convert a data URL to a Blob object
function dataURLToBlob(dataURL) {
	const parts = dataURL.split(';base64,');
	const contentType = parts[0].split(':')[1];
	const byteString = atob(parts[1]);
	const arrayBuffer = new ArrayBuffer(byteString.length);
	const uint8Array = new Uint8Array(arrayBuffer);
	for (let i = 0; i < byteString.length; i++) {
		uint8Array[i] = byteString.charCodeAt(i);
	}
	return new Blob([arrayBuffer], { type: contentType });
}


function getActualCoordinates(event, element, zoomLevel) {
    const rect = element.getBoundingClientRect();
    const offsetX = (event.clientX - rect.left) / zoomLevel;
    const offsetY = (event.clientY - rect.top) / zoomLevel;
    return { x: offsetX, y: offsetY };
}

function loadedImageToBlob(image) {
    return new Promise((resolve, reject) => {
        const canvas = document.createElement('canvas');

        canvas.width = image.width;
        canvas.height = image.height;

        const ctx = canvas.getContext('2d', { willReadFrequently: true });

        image.onload = () => {
            ctx.drawImage(image, 0, 0);

            const dataURL = canvas.toDataURL('image/png', 1);

            const blob = dataURLToBlob(dataURL);
            resolve(blob);  // Resolving the promise with the blob
        };

        image.onerror = () => {
            console.error('Error loading image');
            reject(new Error('Error loading image'));  // Rejecting the promise with an error
        };
    });
}


function prepareRGB(image, backupCanvas, backupCtx) {
	// paste mask data into alpha channel
	backupCtx.drawImage(image, 0, 0, backupCanvas.width, backupCanvas.height);
	const backupData = backupCtx.getImageData(0, 0, backupCanvas.width, backupCanvas.height);

	// refine mask image
	for (let i = 0; i < backupData.data.length; i += 4) {
		if(backupData.data[i+3] == 255)
			backupData.data[i+3] = 0;
		else
			backupData.data[i+3] = 255;

		backupData.data[i] = 0;
		backupData.data[i+1] = 0;
		backupData.data[i+2] = 0;
	}

	backupCtx.globalCompositeOperation = 'source-over';
	backupCtx.putImageData(backupData, 0, 0);
}

class PaintAction {
    constructor(previousImageData, actionType, points, color, canvasType) {
        this.previousImageData = previousImageData;
        this.actionType = actionType;
        this.points = points;
        this.color = color;
        this.canvasType = canvasType;  // 'imgCanvas' or 'maskCanvas'
    }
}

class ComfyShopDialog extends ComfyDialog {
	static instance = null;

	static getInstance() {
		if(!ComfyShopDialog.instance) {
			ComfyShopDialog.instance = new ComfyShopDialog(app);
		}

		return ComfyShopDialog.instance;
	}

	is_layout_created =  false;

	constructor() {
		if(ComfyShopDialog.instance) {
			throw new Error("Use ComfyShopDialog.getInstance() to get an instance of this class.");
		}
		
		super();
		this.eventHandlers = {
			contextmenu: (event) => { event.preventDefault(); },
			wheel: (event) => this.handleWheelEvent(this, event),
			keydown: (event) => this.handleKeyDown(this, event),
			keyup: (event) => this.handleKeyUp(this, event),
			pointerdown: (event) => this.handlePointerDown(this, event),
			pointerup: (event) => ComfyShopDialog.handlePointerUp(event),
			pointermove: (event) => this.draw_move(this, event),
			touchmove: (event) => this.draw_move(this, event),
			pointerout: (event) => this.handlePointerOut(event),
			pointerover: (event) => this.handlePointerOver(event),
			documentPointerMove: (event) => ComfyShopDialog.getInstance().handlePointerMove(event),
			childContainerMouseMove: (event) => this.handlePointerMove(event, this.childContainer)
		};

		this.rgbBlob = null;
        this.rgbItem = null;

		const styleSheet = document.createElement("style");
		styleSheet.type = "text/css";
		styleSheet.innerText = '.brush-preview { transition: opacity 0.1s ease; }'; // Adjust the transition time to a smaller value to reduce flickering
		document.head.appendChild(styleSheet);

		this.isGreyscale = 'greyscale'
		var brush = document.createElement("div");
		brush.id = "brush";
		brush.style.backgroundColor = "transparent";
		brush.style.outline = "1px dashed black";
		brush.style.boxShadow = "0 0 0 1px white";
		brush.style.borderRadius = "50%";
		brush.style.MozBorderRadius = "50%";
		brush.style.WebkitBorderRadius = "50%";
		brush.style.position = "absolute";
		brush.style.zIndex = 8889;
		brush.style.pointerEvents = "none";
		this.brush_softness = 0.1
		this.brushColor = "#000000";  // Default brush color
		this.lastRGBColor = "#000000";
		this.isAltDown = false;
		this.isCtrlDown = false;
		this.isSpaceDown = false;
		this.initialMouseX = null;
		this.initialMouseY = null;
		this.previousMousePosition = { x: 0, y: 0 };
		this.zoomLevel = 1.0;
		this.zoomCenterX = 0;
		this.zoomCenterY = 0;
		this.brush = brush;
		this.brushVisible = true; 
		this.brushOpacity = 1.0; // a value between 0.0 and 1.0
		this.brush_size = 50; // Initialize brush size
        this.contextMenu = this.createContextMenu(); 
		this.isDragging = false;
		this.initialOffsetX = 0;
		this.initialOffsetY = 0;

	    this.lastPointerMoveTime = 0;
    	this.pointerMoveThrottleTime = 16; // roughly 60 fps

		this.handleKeyDown = this.handleKeyDown.bind(this);
		this.handleKeyUp = this.handleKeyUp.bind(this);

		document.removeEventListener('keydown', this.handleKeyDown);
		document.removeEventListener('keyup', this.handleKeyUp);
	
		document.addEventListener('keydown', this.handleKeyDown);
		document.addEventListener('keyup', this.handleKeyUp);

		this.isDrawing = false;

		this.currentAction = null;
        this.rgbActionStack = [];
        this.rgbUndoStack = [];
        this.greyActionStack = [];
        this.greyUndoStack = [];
		

		this.left = "0";
		this.top = "0";

		this.element = $el("div.comfy-modal", { parent: document.body }, 
			[ $el("div.comfy-modal-content", 
				[...this.createButtons()]),
			]);
	}

	createButtons() {
		return [];
	}

	createButton(name, callback) {
		var button = document.createElement("button");
		button.innerText = name;
		button.addEventListener("click", callback);
		return button;
	}

	createLeftButton(name, callback) {
		var button = this.createButton(name, callback);
		button.style.cssFloat = "left";
		button.style.marginRight = "4px";
		return button;
	}

	createRightButton(name, callback) {
		var button = this.createButton(name, callback);
		button.style.cssFloat = "right";
		button.style.marginLeft = "4px";
		return button;
	}

	createContextMenu() {
		// Create a div element for the context menu
		const contextMenu = document.createElement('div');
		contextMenu.id = 'canvasContextMenu';
		contextMenu.style.position = 'absolute';
		contextMenu.style.display = 'none';
		contextMenu.style.zIndex = '99999'; // Set a high z-index value
	
		// Create a div for the dark grey parent box
		const menuBox = document.createElement('div');
		menuBox.style.backgroundColor = '#333'; // Dark grey background color
		menuBox.style.padding = '5px'; // Add some padding for spacing
		menuBox.style.borderRadius = '5px'; // Rounded border
		menuBox.style.border = '2px solid #777'; // 4px light grey border

		const colorPickerWrapper = this.createColorPickerWrapper();
		
		menuBox.appendChild(colorPickerWrapper);
	
		const thicknessSlider = this.createSlider("Thickness", (event) => {
			// Update brush size
			this.brush_size = event.target.value;
			// Handle any additional logic related to brush size change
			
			// Update the brush preview
			this.updateBrushPreview(this);
		}, this.brush_size);
		thicknessSlider.id = "thicknessSlider";
	
		const opacitySlider = this.createSlider("Opacity", (event) => {
			// Update brush opacity (dividing by 100 to get a value between 0 and 1)
			this.brushOpacity = event.target.value / 100;

			// Update the brush preview
			this.updateBrushPreview(this);
		}, this.brushOpacity * 100);  // Note the multiplication here to set the initial slider position
		opacitySlider.id = "opacitySlider";

		// Create a div for the "Softness" slider
		const softnessSlider = this.createSlider("Softness", (event) => {
			// Update brush softness
			const sliderValue = parseFloat(event.target.value); // assuming the slider value is between 0 and 1
			this.brush_softness = sliderValue * 1; // map the slider value to the range [0, 1]
			
			// Handle any additional logic related to brush softness change
			
			// Update the brush preview
			this.updateBrushPreview(this);
		}, this.brush_softness / 1); // normalize the initial brush softness to the slider's range
		softnessSlider.id = "softnessSlider";

		// Append sliders to the dark grey parent box
		menuBox.appendChild(thicknessSlider);
		menuBox.appendChild(opacitySlider);
		menuBox.appendChild(softnessSlider);
	
		// Append the dark grey parent box to the context menu
		contextMenu.appendChild(menuBox);
	
		// Append the context menu to the body or another container (if available)
		document.body.appendChild(contextMenu);
	
		return contextMenu; // Return the context menu element for future reference
	}
	
	createSlider(name, callback, defaultValue) {
		const sliderDiv = document.createElement('div');
	
		// Create a label for the slider
		const labelElement = document.createElement('label');
		labelElement.textContent = name;
	
		// Create the slider input element
		const sliderInput = document.createElement('input');
		sliderInput.setAttribute('type', 'range');
		sliderInput.setAttribute('min', '1');
		sliderInput.setAttribute('max', '100');
		sliderInput.setAttribute('value', defaultValue || '10'); // Set the default value using the provided defaultValue
	
		// Add an event listener to the slider
		sliderInput.addEventListener("input", callback);
	
		// Append label and slider input to the slider div
		sliderDiv.appendChild(labelElement);
		sliderDiv.appendChild(sliderInput);
	
		return sliderDiv;
	}

	createColorPickerWrapper() {
		this.colorPickerWrapper = document.createElement('div');
		this.colorPickerWrapper.id = 'colorPickerWrapper';
	
		this.colorPickerWrapper.style.position = 'relative';
		this.colorPickerWrapper.style.display = 'inline-block';
		this.colorPickerWrapper.style.borderRadius = '5px';
		this.colorPickerWrapper.style.border = '2px solid #777';
		this.colorPickerWrapper.style.overflow = 'hidden';
	
		this.colorPicker = document.createElement('input');
	
		this.colorPicker.type = 'color';
		this.colorPicker.style.borderRadius = '10px';
		this.colorPicker.style.border = 'none';
		this.colorPicker.style.outline = 'none';
		this.colorPicker.style.backgroundColor = 'transparent';
		this.colorPickerWrapper.appendChild(this.colorPicker);
		this.colorPicker.value = this.lastRGBColor || "#000000";
	
		this.colorPickerWrapper.addEventListener('click', (event) => {
			this.colorPicker.click();
		});
	
		this.colorPicker.addEventListener('input', (event) => {
			this.brushColor = event.target.value;
			this.lastRGBColor = this.colorPicker.value;
			this.updateBrushPreview(this);
		});
		
		return this.colorPickerWrapper;
	}

	createComboBox() {
		var comboBox = document.createElement("select");
		comboBox.style.cssFloat = "right";
		comboBox.style.marginLeft = "4px";
		comboBox.style.marginRight = "4px";
		comboBox.style.height = "29px";
	
		var option1 = document.createElement("option");
		option1.value = "greyscale";
		option1.text = "Greyscale";
		comboBox.appendChild(option1);
	
		var option2 = document.createElement("option");
		option2.value = "rgb";
		option2.text = "RGB";
		comboBox.appendChild(option2);
	
		comboBox.addEventListener('change', (event) => {
			if (event.target.value === 'greyscale') {
				this.isGreyscale = 'greyscale';
				this.unbindEventsFromCanvas(this.imgCanvas);
				this.setEventHandler(this.maskCanvas);
				this.maskCanvas.style.visibility = 'visible';
				this.colorPicker.type = 'range';
				this.colorPicker.value = "#000000";
				this.brushColor = "#000000";
				this.colorPickerWrapper.style.backgroundColor = "#000000";
			} else {
				this.isGreyscale = 'rgb';
				this.colorPicker.type = 'color';
				// Switch to the RGB canvas and update the visibility
				this.unbindEventsFromCanvas(this.maskCanvas);
				this.setEventHandler(this.imgCanvas);
				this.maskCanvas.style.visibility = 'hidden';
				this.colorPicker.value = this.lastRGBColor;
				this.brushColor = this.lastRGBColor;
			}
		});
	
		return comboBox;
	}

	setlayout(imgCanvas, maskCanvas) {
		const self = this;
	
		// If it is specified as relative, using it only as a hidden placeholder for padding is recommended
		// to prevent anomalies where it exceeds a certain size and goes outside of the window.
		var placeholder = document.createElement("div");
		placeholder.style.position = "relative";
		placeholder.style.height = "50px";

		// Update brush references to this.brush
		this.brush.id = "brush";
		this.brush.style.backgroundColor = "transparent";
		this.brush.style.outline = "1px dashed black";
		this.brush.style.boxShadow = "0 0 0 1px white";
		this.brush.style.borderRadius = "50%";
		this.brush.style.MozBorderRadius = "50%";
		this.brush.style.WebkitBorderRadius = "50%";
		this.brush.style.position = "absolute";
		this.brush.style.zIndex = 99999;
		this.brush.style.pointerEvents = "none";		
		
		// Create a child container
		this.childContainer = document.createElement("div");
		this.childContainer.id = "childContainer";
		this.childContainer.style.position = "relative";

		// Step 1: Create a wrapper container
		var wrapperDiv = document.createElement("div");
		wrapperDiv.id = "wrapperDiv";
		wrapperDiv.style.position = "relative";
		wrapperDiv.style.overflow = "hidden";
		wrapperDiv.style.maxWidth = "100%";
		wrapperDiv.style.maxHeight = "100%";

		wrapperDiv.style.border = "2px solid #777";
		wrapperDiv.style.borderRadius = "10px";

		var bottom_panel = document.createElement("div");
		bottom_panel.style.position = "absolute";
		bottom_panel.style.bottom = "0px";
		bottom_panel.style.left = "20px";
		bottom_panel.style.right = "20px";
		bottom_panel.style.height = "50px";
		bottom_panel.style.display = "flex";
		bottom_panel.style.justifyContent = "space-between";

		// Append imgCanvas and maskCanvas to the child container
		this.childContainer.appendChild(imgCanvas);
		this.childContainer.appendChild(maskCanvas);
		wrapperDiv.appendChild(this.childContainer);
		this.element.appendChild(wrapperDiv);
		this.element.appendChild(placeholder); // must below z-index than bottom_panel to avoid covering button
		this.element.appendChild(bottom_panel);
		document.body.appendChild(this.brush);

		var clearButton = this.createLeftButton("Clear",
			() => {
				self.maskCtx.clearRect(0, 0, self.maskCanvas.width, self.maskCanvas.height);
				self.backupCtx.clearRect(0, 0, self.backupCanvas.width, self.backupCanvas.height);
			});

		var cancelButton = this.createRightButton("Cancel", () => {
			this.zoomLevel = 1;  // Reset zoom level to 1
			this.childContainer.style.transform = `scale(${this.zoomLevel})`;  // Resetting zoom level in DOM
			this.childContainer.style.left = this.left;
			this.childContainer.style.top = this.top;

			// Hide the context menu
			const contextMenu = document.getElementById('canvasContextMenu');
			contextMenu.style.display = 'none';
			
			self.close();
		});


		this.saveButton = this.createRightButton("Save", () => {
			document.removeEventListener('keydown', this.boundHandleKeyDown);
			document.removeEventListener('keyup', this.boundHandleKeyUp);
			this.maskCanvas.removeEventListener('pointerdown', this.handlePointerDown);
			document.removeEventListener('pointerup', this.handlePointerUp);
			this.zoomLevel = 1;  // Reset zoom level to 1
			this.childContainer.style.transform = `scale(${this.zoomLevel})`;  // Resetting zoom level in DOM
			this.childContainer.removeEventListener('mousemove',  this.handlePointerMove);
			this.childContainer.style.left = this.left;
			this.childContainer.style.top = this.top;

			// Hide the context menu
			const contextMenu = document.getElementById('canvasContextMenu');
			contextMenu.style.display = 'none';

			this.save()
			});

		var leftGroup = document.createElement("div");
		leftGroup.style.display = "flex";
		leftGroup.style.alignItems = "center";
		leftGroup.appendChild(clearButton);

		var rightGroup = document.createElement("div");
		rightGroup.style.display = "flex";
		rightGroup.style.alignItems = "center";
		rightGroup.appendChild(cancelButton);

		var comboBox = this.createComboBox();
		rightGroup.appendChild(comboBox);
		
		rightGroup.appendChild(this.saveButton);

		bottom_panel.appendChild(leftGroup);
		bottom_panel.appendChild(rightGroup);

		imgCanvas.style.position = "relative";
		imgCanvas.style.top = "200";
		imgCanvas.style.left = "0";

		maskCanvas.style.position = "absolute";
	}

	show() {
		if(!this.is_layout_created) {
			// layout
			const imgCanvas = document.createElement('canvas');
			const maskCanvas = document.createElement('canvas');
			const backupCanvas = document.createElement('canvas');

			imgCanvas.id = "imageCanvas";
			maskCanvas.id = "maskCanvas";
			backupCanvas.id = "backupCanvas";

			this.setlayout(imgCanvas, maskCanvas);

			// prepare content
			this.imgCanvas = imgCanvas;
			this.maskCanvas = maskCanvas;
			this.backupCanvas = backupCanvas;
			this.maskCtx = maskCanvas.getContext('2d', { willReadFrequently: true });
			this.backupCtx = backupCanvas.getContext('2d', { willReadFrequently: true });
			
			this.setEventHandler(maskCanvas);

			document.addEventListener('keydown', this.boundHandleKeyDown);
			document.addEventListener('keyup', this.boundHandleKeyUp);



			this.is_layout_created = true;

			const self = this;
			const observer = new MutationObserver(function(mutations) {
			mutations.forEach(function(mutation) {
					if (mutation.type === 'attributes' && mutation.attributeName === 'style') {
						if(self.last_display_style && self.last_display_style != 'none' && self.element.style.display == 'none') {
							ComfyApp.onClipspaceEditorClosed();
						}

						self.last_display_style = self.element.style.display;
					}
				});
			});

			const config = { attributes: true };
			observer.observe(this.element, config);
		}

		this.setImages(this.imgCanvas, this.backupCanvas);

		if(ComfyApp.clipspace_return_node) {
			this.saveButton.innerText = "Save to node";
		}
		else {
			this.saveButton.innerText = "Save";
		}
		this.saveButton.disabled = false;

		this.element.style.display = "block";
		this.element.style.zIndex = 8888; // NOTE: alert dialog must be high priority.
	}

	isOpened() {
		return this.element.style.display == "block";
	}

	resizeHandler = () => {
		// Ensure the image is fully loaded
		if (!this.image.complete) {
			return;
		}
	
		// Calculate the aspect ratio
		const aspectRatio = this.image.width / this.image.height;
	
		// Set the maximum height and calculate the corresponding width
		const maxImgHeight = window.innerHeight * 0.7;
		let containerWidth = maxImgHeight * aspectRatio;
		let containerHeight = maxImgHeight;
	
		// Ensure the width does not exceed the window's width
		if (containerWidth > window.innerWidth) {
			containerWidth = window.innerWidth;
			containerHeight = containerWidth / aspectRatio;
		}
	
		// Set the canvas sizes to fit the container
		this.imgCanvas.width = containerWidth;
		this.imgCanvas.height = containerHeight;
	
		// Resize both mask and backup canvases to the same dimensions
		this.maskCanvas.width = this.backupCanvas.width = containerWidth;
		this.maskCanvas.height = this.backupCanvas.height = containerHeight;

		// Draw the image onto the imgCanvas
		this.imgCtx.clearRect(0, 0, this.imgCanvas.width, this.imgCanvas.height);
		this.imgCtx.drawImage(this.image, 0, 0, this.imgCanvas.width, this.imgCanvas.height);
	
		// Updating the mask and copying between canvases
		this.maskCanvas.style.top = this.imgCanvas.offsetTop + "px";
		this.maskCanvas.style.left = this.imgCanvas.offsetLeft + "px";
		this.backupCtx.drawImage(this.maskCanvas, 0, 0, this.maskCanvas.width, this.maskCanvas.height);
		this.maskCtx.drawImage(this.backupCanvas, 0, 0, this.backupCanvas.width, this.backupCanvas.height);

	};

	setImages(imgCanvas, backupCanvas) {
		this.imgCtx = imgCanvas.getContext('2d', { willReadFrequently: true });
		const backupCtx = backupCanvas.getContext('2d', { willReadFrequently: true });
		const maskCtx = this.maskCtx;
		const maskCanvas = this.maskCanvas;

		let origRetryCount = 0;
		let touchedRetryCount = 0;
		const maxRetries = 3;
	
		backupCtx.clearRect(0, 0, this.backupCanvas.width, this.backupCanvas.height);
		this.imgCtx.clearRect(0, 0, this.imgCanvas.width, this.imgCanvas.height);
		maskCtx.clearRect(0, 0, this.maskCanvas.width, this.maskCanvas.height);
	
		// image load
		const orig_image = new Image();
		window.removeEventListener("resize", this.resizeHandler);
		window.addEventListener("resize", this.resizeHandler);
	
		const touched_image = new Image();
	
		touched_image.onload = function() {
			backupCanvas.width = touched_image.width;
			backupCanvas.height = touched_image.height;
		
			prepareRGB(touched_image, backupCanvas, backupCtx);
		};
		
		touched_image.onerror = (errorEvent) => {
			console.error('Failed to load the touched image:', errorEvent);
			if (touchedRetryCount < maxRetries) {
				touched_image.src = touched_image.src;
				touchedRetryCount++;
			} else {
				alert('We encountered an issue loading the touched image multiple times. Please try again later.');
			}
		};
	
		const alpha_url = new URL(ComfyApp.clipspace.imgs[ComfyApp.clipspace['selectedIndex']].src);
		alpha_url.searchParams.delete('channel');
		alpha_url.searchParams.delete('preview');
		alpha_url.searchParams.set('channel', 'a');
		touched_image.src = alpha_url.href;
	
		// Preloading the image for better perceived performance
		const preloadedImage = new Image();
		preloadedImage.src = alpha_url.href;
	
		orig_image.onload = () => {
			this.originalWidth = orig_image.width;
			this.originalHeight = orig_image.height;

			// Calculate grid size based on image dimensions
			const gridSize = Math.min(
				this.originalWidth / Math.round(this.originalWidth / 20),
				this.originalHeight / Math.round(this.originalHeight / 20)
			);

			// Update the grid background with the new gridSize
			wrapperDiv.style.background = `
			repeating-linear-gradient(
				0deg,
				transparent,
				transparent ${gridSize - 1}px,
				#777 ${gridSize - 1}px,
				#777 ${gridSize}px
			),
			repeating-linear-gradient(
				90deg,
				transparent,
				transparent ${gridSize - 1}px,
				#777 ${gridSize - 1}px,
				#777 ${gridSize}px
			)`;

			window.dispatchEvent(new Event('resize'));
		};
		
		orig_image.onerror = (errorEvent) => {
			console.error('Failed to load the original image:', errorEvent);
			if (origRetryCount < maxRetries) {
				orig_image.src = orig_image.src;
				origRetryCount++;
			} else {
				alert('We encountered an issue loading the original image multiple times. Please try again later.');
			}
		};
	
		const rgb_url = new URL(ComfyApp.clipspace.imgs[ComfyApp.clipspace['selectedIndex']].src);
		rgb_url.searchParams.delete('channel');
		rgb_url.searchParams.set('channel', 'rgb');
		orig_image.src = rgb_url.href;
		this.image = orig_image;
	}

    setEventHandler(canvas) {
        Object.keys(this.eventHandlers).forEach(eventName => {
            if (eventName === 'documentPointerMove') {
                this.childContainer.addEventListener('pointermove', this.eventHandlers[eventName]);
            } else if (eventName === 'childContainerMouseMove') {
                this.childContainer.addEventListener('mousemove', this.eventHandlers[eventName]);
            } else if (eventName === 'wheel') {
				this.childContainer.addEventListener('wheel', this.eventHandlers.wheel, { passive: false });
			} else if (eventName === 'touchmove') {
				this.childContainer.addEventListener('touchmove', this.eventHandlers.touchmove, { passive: true });
			} else {
                canvas.addEventListener(eventName, this.eventHandlers[eventName]);
            }
        });

		document.addEventListener('contextmenu', (event) => {
			event.preventDefault();
			event.stopPropagation(); 
		});
		
		canvas.addEventListener('contextmenu', (event) => {
			if (event.altKey || event.shiftKey) {
			  event.preventDefault();
			  event.stopPropagation(); 
		  
			  const mouseX = event.clientX;
			  const mouseY = event.clientY;
		  
			  const contextMenu = document.getElementById('canvasContextMenu');
			  contextMenu.style.left = mouseX + 'px';
			  contextMenu.style.top = mouseY + 'px';
			  contextMenu.style.display = 'block';
			}
    	});
	}

	unbindEventsFromCanvas(canvas) {
		Object.keys(this.eventHandlers).forEach(eventName => {
			if (eventName === 'documentPointerMove') {
				document.removeEventListener('pointermove', this.eventHandlers[eventName]);
			} else if (eventName === 'childContainerMouseMove') {
				this.childContainer.removeEventListener('mousemove', this.eventHandlers[eventName]);
			} else if (eventName === 'wheel') {
				this.childContainer.removeEventListener('wheel', this.eventHandlers.wheel, { passive: true });
			} else if (eventName === 'touchmove') {
				this.childContainer.removeEventListener('touchmove', this.eventHandlers.touchmove, { passive: true });
			} else {
				canvas.removeEventListener(eventName, this.eventHandlers[eventName]);
			}
		});
	}

	handlePointerOut(event) {
		this.drawing_mode = false;
		this.cursorX = event.pageX;
        this.cursorY = event.pageY;
        this.updateBrushPreview(this);
		this.brush.style.display = 'none';
	}

	handlePointerOver(event) {
		this.brush.style.display = 'block';
		if (event.buttons === 1 || event.buttons === 2) {
			this.drawing_mode = true;
		}
		this.cursorX = event.pageX;
        this.cursorY = event.pageY;
        this.updateBrushPreview(this);
	}

	handlePointerMove(event) {
		if (this.isCtrlDown && this.isSpaceDown) {
			if (event.buttons === 1) {
				if (this.isDragging) {
					this.handleDragMove(event);
				}
				return;  // Prevent zooming when dragging
			}
		
			const currentTime = Date.now();
			if (currentTime - this.lastPointerMoveTime < 16) { 
				return;
			}
		
			if (this.initialMouseX === null) {
				this.initialMouseX = event.clientX; 
				const rect = this.childContainer.getBoundingClientRect();
				const style = window.getComputedStyle(this.childContainer);
				const matrix = new DOMMatrix(style.transform);
				this.zoomCenterX = (event.clientX - rect.left) / matrix.a - parseFloat(style.left);
				this.zoomCenterY = (event.clientY - rect.top) / matrix.d - parseFloat(style.top);
			}
		
			// Adjust to use movementX and movementY for more immediate response to direction changes
			this.adjustZoomLevel(event.movementX > 0 || event.movementY > 0 ? 'in' : 'out');
		
			requestAnimationFrame(() => {
				this.applyZoom();
			});
		
			this.lastPointerMoveTime = currentTime; 
		} else {
			this.initialMouseX = null;
		}
		
		this.cursorX = event.pageX;
		this.cursorY = event.pageY;
		this.updateBrushPreview(this);
	}
	
	handleDragMove(event) {
		if (this.isDragging) {
			const deltaX = event.clientX - this.initialMouseX;
			const deltaY = event.clientY - this.initialMouseY;
		
			this.childContainer.style.left = `${this.initialOffsetX + deltaX}px`;
			this.childContainer.style.top = `${this.initialOffsetY + deltaY}px`;
		}
	}
	
	adjustZoomLevel(direction) {
		if (this.isDragging) {
			return; // Prevent zooming if a drag move is in progress
		}
	
		let zoomStep = 0.05 * this.zoomLevel;  // Increased the zoom step multiplier for greater sensitivity
		if (direction === 'in') {
			this.zoomLevel = Math.min(this.zoomLevel + zoomStep, 5);
		} else if (direction === 'out') {
			this.zoomLevel = Math.max(this.zoomLevel - zoomStep, 0.1);
		}
	}
	
	applyZoom() {
		if (this.isDragging) {
			return; // Prevent zooming if a drag move is in progress
		}
		
		this.childContainer.style.transform = `scale(${this.zoomLevel})`;
		this.childContainer.style.transformOrigin = `${this.zoomCenterX}px ${this.zoomCenterY}px`;
		this.childContainer.style.willChange = 'transform'; // Hint the browser to optimize for transform changes
	}


	brush_size = 10;
	drawing_mode = false;
	lastx = -1;
	lasty = -1;
	lasttime = 0;

	handleKeyDown(event) {
		const self = ComfyShopDialog.instance;
		if(event.key === 'Control') {
			self.isCtrlDown = true;
		}
		if(event.key === ' ') { // Spacebar key event
			self.isSpaceDown = true;
		}
		if(event.key === 'Alt') { // Alt key event
			self.isAltDown = true;
		}
		if(event.key === 'w') { // Spacebar key event
			console.log('w Down');
			console.log('nodeData Available', this.nodeData.input.required.image[0][0]);
		
			this.copyToClipspace();
		}
		if(event.key === 'f') { // reframe key event
			self.zoomLevel = 1;  // Reset zoom level to 1
			self.childContainer.style.transform = `scale(${self.zoomLevel})`;  // Resetting zoom level in DOM
			self.childContainer.style.left = self.left;
			self.childContainer.style.top = self.top;
		}
		if (event.key === ']') {
			self.brush_size = Math.min(self.brush_size+2, 100);
		} else if (event.key === '[') {
			self.brush_size = Math.max(self.brush_size-2, 1);
		} else if(event.key === 'Enter') {
			self.save();
		}

		if(event.key === 'Shift') {
			self.isShiftDown = true;
		}

		if (event.ctrlKey && event.key.toLowerCase() === 'z') {
			if (event.shiftKey) {
				// If Shift is also pressed, redo
				self.redo();
			} else {
				// Otherwise, just undo
				self.undo();
			}
			
			// Prevent the default behavior for Ctrl+Z and Ctrl+Shift+Z
			event.preventDefault();
		} else if (event.ctrlKey && event.key.toLowerCase() === 'y') {
			// Redo for Ctrl+Y
			self.redo();
	
			// Prevent the default behavior for Ctrl+Y
			event.preventDefault();
		}

		self.updateBrushPreview(self);
	}
	
	copyToClipspace() {
		try {
			const maskCtx = this.maskCanvas.getContext('2d', { willReadFrequently: true });
			const backupCtx = this.backupCanvas.getContext('2d', { willReadFrequently: true });

			const maskImageData = maskCtx.getImageData(0, 0, this.maskCanvas.width, this.maskCanvas.height);
			const backupImageData = backupCtx.getImageData(0, 0, this.backupCanvas.width, this.backupCanvas.height);
	
			ComfyApp.clipspace = {
				'maskImageData': maskImageData,
				'backupImageData': backupImageData,
			};
		} catch (error) {
			console.error('Error copying data to clipspace', error);
		}
	}

	handleKeyUp(event) {
		event.preventDefault();
		event.stopPropagation(); 

		const self = ComfyShopDialog.instance;
	
		if (event.key === 'Control') {
			self.isCtrlDown = false;
		}
		
		if (event.key === ' ') {
			self.isSpaceDown = false;
		}

		if (event.key === 'Alt') {
			self.isAltDown = false;
		}

		if(event.key === 'Shift') {
			self.isShiftDown = false;
		}
		
	}

    get currentActionStack() {
        return this.isGreyscale === 'rgb' ? this.rgbActionStack : this.greyActionStack;
    }

    get currentUndoStack() {
        return this.isGreyscale === 'rgb' ? this.rgbUndoStack : this.greyUndoStack;
    }

	undo() {
		if (this.currentActionStack.length === 0) {
			return;
		}
		const actionToUndo = this.currentActionStack.pop();
	
		if (!actionToUndo) {
			return;
		}
	
		if (actionToUndo.previousImageData) {
			const context = actionToUndo.canvasType === 'imgCanvas' ? this.imgCtx : this.maskCtx;
			context.putImageData(actionToUndo.previousImageData, 0, 0);
		}
		this.currentUndoStack.push(actionToUndo);
	}

	redo() {
		if (this.currentUndoStack.length === 0) {
			return;
		}
	
		const actionToRedo = this.currentUndoStack.pop();
	
		const context = actionToRedo.canvasType === 'imgCanvas' ? this.imgCtx : this.maskCtx;
		
		// Use postActionImageData for redos
		if (actionToRedo.postActionImageData) {
			context.putImageData(actionToRedo.postActionImageData, 0, 0);
		}
		
		// Add the action back to currentActionStack
		this.currentActionStack.push(actionToRedo);
	}

	applyAction(action) {
		const context = action.canvasType === 'imgCanvas' ? this.imgCtx : this.maskCtx;
	
		// Implement logic to apply the action here based on its type.
		switch (action.actionType) {
			case 'drawLine':
				this.drawLines(context, action.points, action.color, action.brushSize);
				break;
			// Add more cases for other action types if needed.
		}
	}

	drawLines(context, points, color, brushSize) {
		// Set the drawing style properties
		context.strokeStyle = color;
		context.lineWidth = brushSize;
		context.lineJoin = 'round';
		context.lineCap = 'round';
	
		// Begin the path for drawing
		context.beginPath();
	
		// Move to the first point (the starting point of the line)
		context.moveTo(points[0].x, points[0].y);
	
		// Loop through the points and draw line segments
		for (let i = 1; i < points.length; i++) {
			context.lineTo(points[i].x, points[i].y);
		}
	
		// Stroke the path to draw the lines
		context.stroke();
	
		// Close the path (optional)
		context.closePath();
	}
	
	drawFromAction(action) {
		if (action.actionType === 'drawLine') {
			const ctx = action.canvasType === 'imgCanvas' ? this.imgCanvas.getContext('2d', { willReadFrequently: true }) : this.maskCanvas.getContext('2d', { willReadFrequently: true });
			ctx.strokeStyle = action.color;
			ctx.lineWidth = action.brushSize;
			ctx.lineJoin = "round";
			ctx.lineCap = "round";
			
			ctx.beginPath();
			for (let i = 0; i < action.points.length - 1; i++) {
				ctx.moveTo(action.points[i].x, action.points[i].y);
				ctx.lineTo(action.points[i + 1].x, action.points[i + 1].y);
				ctx.stroke();
			}
		}
	}

	drawLine(points, color, brushSize, canvasType) {
		if (points.length === 0) {
			return;
		}
	
		// Determine the drawing context based on the canvasType parameter
		const drawingContext = canvasType === 'imgCanvas' ? this.imgCtx : this.maskCtx;
	
		// Set up the drawing context
		drawingContext.globalCompositeOperation = 'source-over';
		drawingContext.imageSmoothingEnabled = true;
		drawingContext.imageSmoothingQuality = 'high';
		drawingContext.lineJoin = 'round';
		drawingContext.lineCap = 'round';
		drawingContext.strokeStyle = color;
		drawingContext.lineWidth = brushSize;
	
		// Begin a new path
		drawingContext.beginPath();
	
		// Move to the first point
		drawingContext.moveTo(points[0].x, points[0].y);
	
		// Draw lines to subsequent points
		for (let i = 1; i < points.length; i++) {
			drawingContext.lineTo(points[i].x, points[i].y);
		}
	
		// Stroke the path
		drawingContext.stroke();
	}

	static handlePointerUp(event) {
		event.preventDefault();
	
		const instance = ComfyShopDialog.instance;
	
		instance.drawing_mode = false;
	
		const drawingContext = instance.isGreyscale === 'rgb'
			? instance.imgCanvas.getContext('2d', { willReadFrequently: true })
			: instance.maskCanvas.getContext('2d', { willReadFrequently: true });
	
		drawingContext.globalAlpha = 1;
	
		if (instance.isDragging) {
			instance.isDragging = false;
			instance.initialMouseX = null; // Reset the initial mouse coordinates
			instance.initialMouseY = null;
		}
	
		if (ComfyShopDialog.instance.currentUndoStack.length > 0) {
			ComfyShopDialog.instance.currentUndoStack.splice(0, ComfyShopDialog.instance.currentUndoStack.length);
		}
	
		// Capture the postActionImageData
		if (instance.currentAction) {
			instance.currentAction.postActionImageData = drawingContext.getImageData(0, 0, 
				(instance.isGreyscale === 'rgb' ? instance.imgCanvas : instance.maskCanvas).width, 
				(instance.isGreyscale === 'rgb' ? instance.imgCanvas : instance.maskCanvas).height);
		}
	
		if (instance.currentAction) {
	
			instance.currentActionStack.push(instance.currentAction);

			instance.currentAction = null;  // Reset the current action for the next one
		}
	}

	updateBrushPreview(self) {
		requestAnimationFrame(() => {
			const brush = self.brush;
		
			var centerX = self.cursorX;
			var centerY = self.cursorY;
		
			const brushColorRgb = this.hexToRgb(self.brushColor); 
	
			brush.style.width = self.brush_size * 2 + "px";
			brush.style.height = self.brush_size * 2 + "px";
			brush.style.left = (centerX - self.brush_size) + "px";
			brush.style.top = (centerY - self.brush_size) + "px";
			brush.style.borderRadius = "50%"; 
			brush.style.background = `rgba(${brushColorRgb.r}, ${brushColorRgb.g}, ${brushColorRgb.b}, ${self.brushOpacity})`; 
			brush.style.boxShadow = `0 0 ${self.brush_softness}px rgba(${brushColorRgb.r}, ${brushColorRgb.g}, ${brushColorRgb.b}, ${self.brush_softness})`;
			brush.style.transition = 'top 0.01s, left 0.01s';
			brush.style.transform = 'translate3d(0, 0, 0) scale(1)';
			brush.style.boxSizing = 'border-box';
		
			// Hide the brush preview while dragging or zooming
			if (self.isDragging || self.isZooming) {
				brush.style.visibility = 'hidden';
			} else {
				brush.style.visibility = 'visible';
			}
		});
	}

	handleWheelEvent(self, event) {
		// Target the actual <input> sliders inside the divs by using querySelector
		const thicknessSlider = document.querySelector("#thicknessSlider input[type='range']");
		const opacitySlider = document.querySelector("#opacitySlider input[type='range']");
		const softnessSlider = document.querySelector("#softnessSlider input[type='range']");
	
		if (self.isShiftDown && self.isAltDown) {
			// Adjust the brush opacity
			if (event.deltaY > 0) {
				self.brushOpacity = Math.min(self.brushOpacity + 0.05, 1);
			} else {
				self.brushOpacity = Math.max(self.brushOpacity - 0.05, 0);
			}
			if (opacitySlider) { 
				opacitySlider.value = self.brushOpacity * 100;  // Note the multiplication here
			}
			self.updateBrushPreview(self);
			event.preventDefault();
		} else if (self.isAltDown) { 
			// Adjust the brush softness
			const wheelAdjustment = 0.01;
			if(event.deltaY < 0) {
				self.brush_softness = Math.min(self.brush_softness + wheelAdjustment * 50, 50);
			} else {
				self.brush_softness = Math.max(self.brush_softness - wheelAdjustment * 50, 0); 
			}
			if (softnessSlider) { 
				softnessSlider.value = `${self.brush_softness / 50 * 100}`; // Convert to string and assuming the max value is 100
			}
			
			event.preventDefault(); 
		} else {
			// Adjust brush size
			if(event.deltaY < 0) {
				self.brush_size = Math.min(self.brush_size + 2, 100);
			} else {
				self.brush_size = Math.max(self.brush_size - 2, 1);
			}
			if (thicknessSlider) {
				thicknessSlider.value = `${self.brush_size}`; // Convert to string
			}
			event.preventDefault();
		}
	
		self.updateBrushPreview(self);
	}

	hexToRgb(hex) {
		let result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
		return result ? {
			r: parseInt(result[1], 16),
			g: parseInt(result[2], 16),
			b: parseInt(result[3], 16)
		} : { r: 0, g: 0, b: 0 }; 
	}

	draw_move(self, event) {
		if (event) {
			event.preventDefault();
		}
	
		const drawingContext = this.isGreyscale === 'rgb' ? self.imgCtx : self.maskCtx;
		const { x, y } = getActualCoordinates(event, self.maskCanvas, self.zoomLevel);
		let brush_size = this.brush_size / self.zoomLevel;
	
		if (this.isCtrlDown && this.isSpaceDown && event.buttons === 1) {
			return;
		}
	
		if (!self.drawing_mode) {
			return;
		}
	
		if (event instanceof PointerEvent && event.pointerType == 'pen') {
			brush_size *= event.pressure;
			this.last_pressure = event.pressure;
		} else if (window.TouchEvent && event instanceof TouchEvent) {
			brush_size *= this.last_pressure;
		}
	
		const brushColorRgb = this.hexToRgb(this.brushColor);
	
		if (this.brushVisible) {
			this.updateBrushPreview(self);
		}
	
		this.cursorX = event.pageX;
		this.cursorY = event.pageY;
		self.updateBrushPreview(self);
	
		const operation = event.buttons === 1 ? "source-over" : event.buttons === 2 ? "destination-out" : null;
		const currentBrushOpacity = this.brushOpacity;
	
		const softFactor = Math.max(0.1, 1 - self.brush_softness / 100); // Normalize brush_softness to [0,1]
		const innerRadius = brush_size * softFactor;
		const outerRadius = brush_size;

		const gradient = drawingContext.createRadialGradient(x, y, innerRadius, x, y, outerRadius);
		
		if (operation === "destination-out") {
			gradient.addColorStop(0, `rgba(255, 255, 255, 1)`);
			gradient.addColorStop(1 - softFactor, `rgba(255, 255, 255, ${currentBrushOpacity * softFactor})`);
			gradient.addColorStop(1, `rgba(255, 255, 255, 0)`); // This ensures the outer edge is fully transparent
		} else {
			gradient.addColorStop(0, `rgba(${brushColorRgb.r}, ${brushColorRgb.g}, ${brushColorRgb.b}, ${currentBrushOpacity})`);
			gradient.addColorStop(1 - softFactor, `rgba(${brushColorRgb.r}, ${brushColorRgb.g}, ${brushColorRgb.b}, ${currentBrushOpacity * softFactor})`);
			gradient.addColorStop(1, `rgba(${brushColorRgb.r}, ${brushColorRgb.g}, ${brushColorRgb.b}, 0)`); // This ensures the outer edge is fully transparent
		}
		
		if (operation && !this.isDrawing) {
			this.isDrawing = true;


			requestAnimationFrame(() => {
				drawingContext.globalCompositeOperation = operation;
				drawingContext.imageSmoothingEnabled = true;
				drawingContext.imageSmoothingQuality = 'high';
				drawingContext.lineJoin = 'round';
				drawingContext.lineCap = 'round';


				if (self.lastx !== null && self.lasty !== null) {
					const distance = Math.hypot(x - self.lastx, y - self.lasty);
					const pointsToFill = Math.ceil(distance / 0.01);

					drawingContext.globalAlpha = this.brushOpacity; 
					drawingContext.strokeStyle = gradient;
					drawingContext.beginPath();
					drawingContext.moveTo(self.lastx, self.lasty);
	
					const controlPoint1 = { x: (self.lastx + x) / 2, y: self.lasty };
					const controlPoint2 = { x: (self.lastx + x) / 2, y: y };
	
					drawingContext.bezierCurveTo(controlPoint1.x, controlPoint1.y, controlPoint2.x, controlPoint2.y, x, y);
					drawingContext.lineWidth = brush_size * 2;
					drawingContext.stroke();
	
					self.lastx = x;
					self.lasty = y;
	
					this.currentAction = {
						actionType: 'drawLine',
						points: [ { x: self.lastx, y: self.lasty }, { x, y } ],
						color: gradient,
						brushSize: brush_size,
						canvasType: this.isGreyscale === 'rgb' ? 'imgCanvas' : 'maskCanvas',
						previousImageData: this.previousImageData
					};
				}
				self.lasttime = performance.now();
				this.isDrawing = false;
			});
		}
	}
	
	handlePointerDown(self, event) {
		let skipDrawing = false;
		// Hide the context menu before any drawing operation
		this.contextMenu.style.display = 'none';
		
		// Check for Alt + right mouse button click to show/hide context menu on mouse down
		if ((event.altKey || event.shiftKey) && event.button === 2 && event.type === 'mousedown') {
			event.preventDefault();
			event.stopPropagation();
		
			const mouseX = event.clientX;
			const mouseY = event.clientY;
		
			// Toggle the context menu visibility
			if (this.contextMenu.style.display === 'block') {
				// Hide the context menu
				this.contextMenu.style.display = 'none';
			} else {
				// Show the context menu
				this.contextMenu.style.left = mouseX + 'px';
				this.contextMenu.style.top = mouseY + 'px';
				this.contextMenu.style.display = 'block';
		
				this.createContextMenu();
			}
		
			// Prevent drawing when context menu is displayed
			skipDrawing = true;
		}
		
		if (!skipDrawing) {
			if (this.isCtrlDown && this.isSpaceDown && event.buttons === 1) {
				this.isDragging = true;
		
				this.initialMouseX = event.clientX;
				this.initialMouseY = event.clientY;
		
				this.initialOffsetX = this.childContainer.offsetLeft;
				this.initialOffsetY = this.childContainer.offsetTop;
		
				// Set drawing_mode to false when Ctrl+Spacebar are pressed and dragging
				this.drawing_mode = false;
		
				return;  // prevent drawing when Ctrl+Spacebar are pressed and dragging
			}
		
			const drawingContext = this.isGreyscale === 'rgb'
				? this.imgCanvas.getContext('2d', { willReadFrequently: true })
				: this.maskCanvas.getContext('2d', { willReadFrequently: true });
		
			// Always capture the current state of the canvas for undo functionality.
			this.previousImageData = drawingContext.getImageData(0, 0, (this.isGreyscale === 'rgb' ? this.imgCanvas : this.maskCanvas).width, (this.isGreyscale === 'rgb' ? this.imgCanvas : this.maskCanvas).height);
		
			const { x, y } = getActualCoordinates(event, this.isGreyscale === 'rgb' ? this.imgCanvas : this.maskCanvas, self.zoomLevel);
		
			var brush_size = this.brush_size / self.zoomLevel;
			if (event instanceof PointerEvent && event.pointerType == 'pen') {
				brush_size *= event.pressure;
				this.last_pressure = event.pressure;
			} else if (window.TouchEvent && event instanceof TouchEvent) {
				brush_size *= this.last_pressure;
			}
		
			const ColorContext = this.isGreyscale === 'rgb' ? self.brushColor : '#000000';
			const brushColorRgb = this.hexToRgb(ColorContext);
			const currentBrushOpacity = this.brushOpacity; 
		
			const softFactor = Math.max(0.1, 1 - self.brush_softness);
			const innerRadius = brush_size * softFactor;
			const outerRadius = brush_size;
		
			const gradient = drawingContext.createRadialGradient(x, y, innerRadius, x, y, outerRadius);
			gradient.addColorStop(0, `rgba(${brushColorRgb.r}, ${brushColorRgb.g}, ${brushColorRgb.b}, ${currentBrushOpacity})`);
			gradient.addColorStop(1, `rgba(${brushColorRgb.r}, ${brushColorRgb.g}, ${brushColorRgb.b}, ${currentBrushOpacity * softFactor})`);
		
			drawingContext.beginPath();
			drawingContext.arc(x, y, brush_size, 0, 2 * Math.PI);
			drawingContext.fillStyle = gradient;
			drawingContext.fill();
		
			self.lastx = x;
			self.lasty = y;
		
			this.drawing_mode = true;
			this.brushVisible = true;
		
			this.draw_move(this, event);

			this.currentAction = {
				actionType: 'drawDot',
				points: [ { x, y } ],
				color: gradient,
				brushSize: brush_size,
				canvasType: this.isGreyscale === 'rgb' ? 'imgCanvas' : 'maskCanvas',
				previousImageData: this.previousImageData
			};
		}
	}

	async saveRGBImage() {
		const imgFilename = "clipspace-rgba-" + performance.now() + ".png";
		try {
			const self = this;
	
			const offscreen = new OffscreenCanvas(this.originalWidth, this.originalHeight);
			const offscreenCtx = offscreen.getContext('2d', { willReadFrequently: true });
	
			offscreenCtx.drawImage(this.imgCanvas, 0, 0, this.imgCanvas.width, this.imgCanvas.height, 0, 0, this.originalWidth, this.originalHeight);
			
			if (typeof this.originalWidth !== 'number' || typeof this.originalHeight !== 'number') {
				throw new Error("Original image dimensions are not properly set.");
			}
	
			// Image Processing and Uploading Promises
			let imageProcessingPromise = new Promise((resolve, reject) => {
				offscreen.convertToBlob({type: 'image/png'}).then(blob => { 
	
					const imgFile = new File([blob], imgFilename, { type: "image/png" });
					const original_ref = URL.createObjectURL(blob);
	
					// Image processing for RGB
					const imgFormData = new FormData();
	
					const imgForm = {
						"filename": imgFilename,
						"subfolder": "clipspace",
						"type": "input",
					};
	
					if(ComfyApp.clipspace.images)
						ComfyApp.clipspace.images[0] = imgForm;
			
					if(ComfyApp.clipspace.widgets) {
						const index1 = ComfyApp.clipspace.widgets.findIndex(obj => obj.name === 'image');
			
						if(index1 >= 0)
							ComfyApp.clipspace.widgets[index1].value = imgForm;
					}
	
					imgFormData.append('image', imgFile, imgFilename);
					imgFormData.append('type', "input");
					imgFormData.append('subfolder', "clipspace");
	
					// Create a new image element and set its src attribute to the object URL of the Blob
					const image = new Image();
					image.src = original_ref;
					image.onload = async () => {
						try {
							await self.processAndUploadImage(imgFile, imgForm, imgFormData);
							resolve({ original_ref, imageProcessingPromise, imgFilename });
						} catch (error) {
							console.error('Error converting RGB image to blob:', error);
							reject(error);
						} finally {
							URL.revokeObjectURL(original_ref); // Revoke the blob URL here
						}
					};

					image.onerror = (error) => {
						console.error('RGB Image OnLoad Error:', error);
						reject(error);
						URL.revokeObjectURL(original_ref); // Revoke the blob URL here as well
					};
	
				}).catch(error => {
					console.error('Error converting RGB offscreen canvas to blob:', error);
					reject(error);
				});
			});
	
			return imageProcessingPromise;
	
		} catch (error) {
			console.error('Error in saving RGB image:', error);
			throw error;
		}
	}


	async save() {
		try {
			const { imgFilename } = await this.saveRGBImage();
			ComfyApp.onClipspaceEditorSave();
			await this.save_mask(imgFilename); 
		} catch (error) {
			console.error('Error in save function:', error);
		}
	}
	
	async processAndUploadImage(image, item, formData, isMask = false) {
		try {
			formData.append('image', image, item.filename);  
	
			this.saveButton.innerText = "Saving...";
			this.saveButton.disabled = true;
	
			// Upload image and return the promise
			return await this.uploadMask(item, formData, isMask);  // Passing isMask to indicate whether this is a mask
		} catch (error) {
			console.error("Error in processAndUploadImage function:", error);
	
			this.saveButton.innerText = "Save";  // Reset button text
			this.saveButton.disabled = false;  // Enable the button again
	
			throw error;  // Propagate the error up the call stack
		}
	}

	async uploadMask(filepath, formData, isMask) {
		const uploadUrl = isMask ? '/upload/mask' : '/upload/image';
		
		try {
			const response = await api.fetchApi(uploadUrl, {
				method: 'POST',
				body: formData
			});
	
			if (response && response.ok) {
				ComfyApp.clipspace.imgs[ComfyApp.clipspace['selectedIndex']] = new Image();
				ComfyApp.clipspace.imgs[ComfyApp.clipspace['selectedIndex']].src = api.apiURL("/view?" + new URLSearchParams(filepath).toString() + app.getPreviewFormatParam());

				if (ComfyApp.clipspace.images) {
					ComfyApp.clipspace.images[ComfyApp.clipspace['selectedIndex']] = filepath;
				} else {
				}
	
				ClipspaceDialog.invalidatePreview();
			} else {
				throw new Error('Unexpected response from the server');
			}
		} catch (error) {
			console.error('Error:', error);
	
			// Cleanup code
			formData = null; // Allow formData to be garbage collected
			if (ComfyApp.clipspace.imgs && ComfyApp.clipspace.imgs[ComfyApp.clipspace['selectedIndex']]) {
				ComfyApp.clipspace.imgs[ComfyApp.clipspace['selectedIndex']].src = ''; // Release the reference to the image URL
				ComfyApp.clipspace.imgs[ComfyApp.clipspace['selectedIndex']] = null; // Allow the Image object to be garbage collected
			}
	
			throw error;  // Re-throw the error to propagate it up the call stack
		}
	}

	async save_mask(imgFilenameFromRGB) {
		const offscreen = new OffscreenCanvas(this.originalWidth, this.originalHeight);
		const offscreenCtx = offscreen.getContext('2d', { willReadFrequently: true });
	
		const original_ref = { filename: imgFilenameFromRGB };
	
		offscreenCtx.drawImage(this.maskCanvas, 0, 0, this.maskCanvas.width, this.maskCanvas.height, 0, 0, offscreen.width, offscreen.height);
	
		// paste mask data into alpha channel
		const imageData = offscreenCtx.getImageData(0, 0, offscreen.width, offscreen.height);
		
		// Create a worker instance
		const worker = new Worker("../../extensions/ComfyI2I/imageProcessorWorker.js");
	
		// Sending imageData to worker
		worker.postMessage({ imageData });
	
		worker.onmessage = async (e) => {
			// Receive the processed data back from the worker
			const processedData = e.data.processedData;
	
			// Put the processed data back onto the offscreen canvas
			offscreenCtx.putImageData(processedData, 0, 0);	
	
			const formData = new FormData();
			const filename = "clipspace-mask-" + performance.now() + ".png";
	
			const item = {
				"filename": filename,
				"subfolder": "clipspace",
				"type": "input",
			};
	
			if(ComfyApp.clipspace.images)
				ComfyApp.clipspace.images[0] = item;

			if(ComfyApp.clipspace.widgets) {
				const index = ComfyApp.clipspace.widgets.findIndex(obj => obj.name === 'image');

				if(index >= 0)
					ComfyApp.clipspace.widgets[index].value = item;
			}

			const blob = await offscreen.convertToBlob({ type: 'image/png' });

			// You can manually set these values or pass them as parameters to the function
			let original_subfolder = "clipspace";
			if(original_subfolder)
				original_ref.subfolder = original_subfolder;

			let original_type = "input";
			if(original_type)
				original_ref.type = original_type;

			formData.append('image', blob, filename);
			formData.append('original_ref', JSON.stringify(original_ref));
			formData.append('type', "input");
			formData.append('subfolder', "clipspace");
		
			this.saveButton.innerText = "Saving...";
			this.saveButton.disabled = true;

			try {
				await this.uploadMask(item, formData, true);  // Ensure this is awaited
				ComfyApp.onClipspaceEditorSave();
				this.close();
			} catch (error) {
				console.error('Error in save_mask function:', error);
				this.saveButton.innerText = "Save";  // Revert the button text
				this.saveButton.disabled = false;  // Re-enable the save button
			}
	
			// Terminate the worker after use
			worker.terminate();
		};

		worker.onerror = (error) => {
			console.error('Error in worker:', error);
			// Handle the error appropriately here
			// Possibly include cleanup code and UI updates to indicate the error to the user
		};
	}
}


app.registerExtension({
	name: "Comfy.ComfyI2I.ComfyShop",
	init(app) {
		const callback =
			function () {
				let dlg = ComfyShopDialog.getInstance();
				dlg.show();
			};

		const context_predicate = () => ComfyApp.clipspace && ComfyApp.clipspace.imgs && ComfyApp.clipspace.imgs.length > 0
		ClipspaceDialog.registerButton("ComfyShop", context_predicate, callback);
	},

	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.output.includes("MASK") && nodeData.output.includes("IMAGE")) {
			addMenuHandler(nodeType, function (_, options) {
				options.unshift({
					content: "Open in ComfyShop",
					callback: () => {
						ComfyApp.copyToClipspace(this);
						ComfyApp.clipspace_return_node = this;

						let dlg = ComfyShopDialog.getInstance();
						dlg.show();
					},
				});
			});
		}
	}
});
