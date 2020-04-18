const MOBILENET_MODEL_PATH = "./model/model.json";
const IMAGE_SIZE = 224;
let mobilenet;

async function load_model() {
  mobilenet = await tf.loadLayersModel(MOBILENET_MODEL_PATH);

  // Warmup the model. Call `dispose` to release the WebGL memory allocated
  mobilenet.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();
  
  // Unhide upload form
  const form = document.getElementById("upload")
  form.style.display = ""

  const result = document.getElementById("result");
  result.innerText = "Let's find out what kind of pet person are you...";
  

};

async function predict(imgElement) {
  const prediction = tf.tidy(() => {
    const img = tf.browser.fromPixels(imgElement).toFloat();
    const normalized = img.div(tf.scalar(255)); // Normalize image to [0, 1].
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]); // Reshape to a single-element batch.
    let prediction = mobilenet.predict(batched);
    return prediction;
  });

  cat_prob = prediction.dataSync()[0];
  const result = document.getElementById("result");
  if (cat_prob >= 0.5) {
    display_prob = (cat_prob * 100).toFixed();
    category = "cat";
  } else {
    display_prob = ((1 - cat_prob) * 100).toFixed();
    category = "dog";
  }
  text = `I'm <span class="result-text">${display_prob}%</span> confident you're a <span class="result-text">${category}</span>...person`;
  result.innerHTML = text;
  return prediction;
};

async function load_image_processor(){
  const filesElement = document.getElementById('file-input');
  filesElement.addEventListener('change', evt => {
    let file = evt.target.files[0];
  // Display thumbnails & issue call to predict each image.
    if (!file.type.match('image.*')) {
      alert("File is not an image");
      return;
    }
    let reader = new FileReader();
    reader.onload = e => {
      // Fill the image & call predict.
      let img = document.createElement('img');
      img.src = e.target.result;
      img.width = IMAGE_SIZE;
      img.height = IMAGE_SIZE;
      img.onload = () => predict(img);
      document.getElementById("img-upload").src = e.target.result;
    };
    // Read in the image file as a data URL.
    reader.readAsDataURL(file);
  });
};

load_model();
load_image_processor();