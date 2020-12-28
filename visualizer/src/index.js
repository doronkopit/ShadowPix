import React, { Component } from "react";
import ReactDOM from "react-dom";
import * as THREE from 'three';
import {OrbitControls} from "three/examples/jsm/controls/OrbitControls";
import {OBJLoader} from "three/examples/jsm/loaders/OBJLoader";

const style = {
    height: 1200 // we can control scene size by setting container dimensions
};

class App extends Component {
    componentDidMount() {
        this.sceneSetup();
        this.addLights();
        this.loadTheModel();
        this.startAnimationLoop();
        window.addEventListener('resize', this.handleWindowResize);
    }

    componentWillUnmount() {
        window.removeEventListener('resize', this.handleWindowResize);
        window.cancelAnimationFrame(this.requestID);
        this.controls.dispose();
    }

    // Standard scene setup in Three.js. Check "Creating a scene" manual for more information
    // https://threejs.org/docs/#manual/en/introduction/Creating-a-scene
    sceneSetup = () => {
        // get container dimensions and use them for scene sizing
        const width = this.mount.clientWidth;
        const height = this.mount.clientHeight;

        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(
            75, // fov = field of view
            width / height, // aspect ratio
            0.1, // near plane
            1000 // far plane
        );
        this.camera.position.z = 500; // is used here to set some distance from a cube that is located at z = 0
        // OrbitControls allow a camera to orbit around the object
        // https://threejs.org/docs/#examples/controls/OrbitControls
        this.controls = new OrbitControls( this.camera, this.mount );
        this.renderer = new THREE.WebGLRenderer();
        this.renderer.setSize( width, height );
        this.mount.appendChild( this.renderer.domElement ); // mount using React ref
    };

    // Code below is taken from Three.js OBJ Loader example
    // https://threejs.org/docs/#examples/en/loaders/OBJLoader
    loadTheModel = () => {
        // instantiate a loader
        const loader = new OBJLoader();

        // load a resource
        loader.load(
            // resource URL relative to the /public/index.html of the app
            'local_output333.obj',
            // called when resource is loaded
            ( object ) => {
                this.scene.add( object );

                // get the newly added object by name specified in the OBJ model (that is Elephant_4 in my case)
                // you can always set console.log(this.scene) and check its children to know the name of a model
                const el = this.scene.getObjectByName("local");

                // change some custom props of the element: placement, color, rotation, anything that should be
                // done once the model was loaded and ready for display
                el.position.set(-200, 250, 0);
                //el.material.color.set(0x50C878);
                el.rotation.z = 4.72;
                //el.rotation.x = -5;
                el.geometry.scale(2, 2, 2);

                // make this element available inside of the whole component to do any animation later
                this.model = el;
            },
            // called when loading is in progresses
             ( xhr ) => {

                const loadingPercentage = Math.ceil(xhr.loaded / xhr.total * 100);
                console.log( ( loadingPercentage ) + '% loaded' );

                // update parent react component to display loading percentage
                this.props.onProgress(loadingPercentage);
            },
            // called when loading has errors
             ( error ) => {

                console.log( 'An error happened:' + error );
                
            }
        );
    };

    // adding some lights to the scene
    addLights = () => {
        const lights = [];

        // set color and intensity of lights
        lights[ 0 ] = new THREE.SpotLight( 0xffffff, 1, 0 );
        lights[0].castShadow = true;
        lights[ 1 ] = new THREE.DirectionalLight( 0xffffff, 1 );
        lights[1].castShadow = true;
        lights[ 2 ] = new THREE.DirectionalLight( 0xffffff, 1 );
        lights[2].castShadow = true;

        // place some lights around the scene for best looks and feel
        //lights[ 0 ].position.set( -600, 0, 350 );
        lights[ 1 ].position.set( 300, 150, 300);
        //lights[1].target.position.set(100, 0, 0);
        //lights[ 2 ].position.set( 400, 0, 300); //Trump
        //lights[ 2 ].position.set( -400, 0, 300); //Lebron
        lights[ 2 ].position.set( 0, 700, 400); //Obama
        //lights[2].target.position.set(100, 0, 0);

        //this.scene.add( lights[ 0 ] );
        //this.scene.add( lights[ 1 ] );
        //this.scene.add(lights[1].target);
        this.scene.add(new THREE.AmbientLight( 0x404040, 0.05 ));
        this.scene.add( lights[2]);
        this.scene.add(lights[2].target);
        let spotLightHelper = new THREE.SpotLightHelper( lights[0]);
        //this.scene.add( spotLightHelper );
        spotLightHelper = new THREE.DirectionalLightHelper( lights[1]);
        //this.scene.add( spotLightHelper );
        spotLightHelper = new THREE.DirectionalLightHelper( lights[2]);
        this.scene.add( spotLightHelper );

        const axesHelper = new THREE.AxesHelper( 5000 );
        this.scene.add( axesHelper );

        const material = new THREE.LineBasicMaterial({
            color: 0x0000ff
        });
        
        const points = [];
        points.push( new THREE.Vector3( - 100, 0, 0 ) );
        points.push( new THREE.Vector3( 100, 0, 0 ) );
        
        const geometry = new THREE.BufferGeometry().setFromPoints( points );
        
        const line = new THREE.Line( geometry, material );
        this.scene.add( line );
    };

    startAnimationLoop = () => {
        // slowly rotate an object
        //if (this.model) this.model.rotation.z += 0.005;

        this.renderer.render( this.scene, this.camera );

        // The window.requestAnimationFrame() method tells the browser that you wish to perform
        // an animation and requests that the browser call a specified function
        // to update an animation before the next repaint
        this.requestID = window.requestAnimationFrame(this.startAnimationLoop);
    };

    handleWindowResize = () => {
        const width = this.mount.clientWidth;
        const height = this.mount.clientHeight;

        this.renderer.setSize( width, height );
        this.camera.aspect = width / height;

        // Note that after making changes to most of camera properties you have to call
        // .updateProjectionMatrix for the changes to take effect.
        this.camera.updateProjectionMatrix();
    };

    render() {
        return <div style={style} ref={ref => (this.mount = ref)} />;
    }
}

class Container extends React.Component {
    state = {isMounted: true};

    render() {
        const {isMounted = true, loadingPercentage = 0} = this.state;
        return (
            <>
                <button onClick={() => this.setState(state => ({isMounted: !state.isMounted}))}>
                    {isMounted ? "Unmount" : "Mount"}
                </button>
                {isMounted && <App onProgress={loadingPercentage => this.setState({ loadingPercentage })} />}
                {isMounted && loadingPercentage === 100 && <div>Scroll to zoom, drag to rotate</div>}
                {isMounted && loadingPercentage !== 100 && <div>Loading Model: {loadingPercentage}%</div>}
            </>
        )
    }
}

const rootElement = document.getElementById("root");
ReactDOM.render(<Container />, rootElement);

/** 
// Define the standard global variables
var container,
  scene,
  camera,
  renderer,
  plane,
  mouseMesh,
  light;

// Custom global variables
var mouse = {
  x: 0,
  y: 0
};

init();
animate();

function init() {

  // Scene
  scene = new THREE.Scene();

  window.addEventListener('resize', function() {
    var WIDTH = window.innerWidth,
      HEIGHT = window.innerHeight;
    renderer.setSize(WIDTH, HEIGHT);
    camera.aspect = WIDTH / HEIGHT;
    camera.updateProjectionMatrix();
  });

  // Camera
  var screenWidth = window.innerWidth,
    screenHeight = window.innerHeight,
    viewAngle = 75,
    nearDistance = 0.1,
    farDistance = 1000;

  camera = new THREE.PerspectiveCamera(viewAngle, screenWidth / screenHeight, nearDistance, farDistance);
  scene.add(camera);
  camera.position.set(0, 0, 5);
  camera.lookAt(scene.position);

  // Renderer engine together with the background
  renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true
  });
  renderer.setSize(screenWidth, screenHeight);
  container = document.getElementById('container');
  container.appendChild(renderer.domElement);

  // Define the lights for the scene
  light = new THREE.PointLight(0xff00ff);
  light.position.set(0, 0, 15);
  scene.add(light);
  var lightAmb = new THREE.AmbientLight(0x000000);
  scene.add(lightAmb);

  // Create a circle around the mouse and move it
  // The sphere has opacity 0
  var mouseGeometry = new THREE.SphereGeometry(1, 100, 100);
  var mouseMaterial = new THREE.MeshLambertMaterial({});
  mouseMesh = new THREE.Mesh(mouseGeometry, mouseMaterial);

  mouseMesh.position.set(0, 0, 0);
  scene.add(mouseMesh);

  // When the mouse moves, call the given function
  document.addEventListener('mousemove', onMouseMove, false);
}

// Follows the mouse event
function onMouseMove(event) {

  // Update the mouse variable
  event.preventDefault();
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

  // Make the sphere follow the mouse
  var vector = new THREE.Vector3(mouse.x, mouse.y, 0.5);
  vector.unproject(camera);
  var dir = vector.sub(camera.position).normalize();
  var distance = -camera.position.z / dir.z;
  var pos = camera.position.clone().add(dir.multiplyScalar(distance));
  //mouseMesh.position.copy(pos);

  light.position.copy(new THREE.Vector3(pos.x, pos.y, pos.z + 2));
};

// Animate the elements
function animate() {
  requestAnimationFrame(animate);
  render();
}

// Rendering function
function render() {

  // For rendering
  renderer.autoClear = false;
  renderer.clear();
  renderer.render(scene, camera);
};**/