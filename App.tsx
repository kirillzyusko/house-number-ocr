import React, { PureComponent } from 'react'
import {
    StyleSheet,
    Text,
    View,
    ActivityIndicator,
    StatusBar,
    Image,
    TouchableOpacity
} from 'react-native'
import * as tf from '@tensorflow/tfjs'
import { fetch, bundleResourceIO, decodeJpeg } from '@tensorflow/tfjs-react-native'
import * as mobilenet from '@tensorflow-models/mobilenet'
import * as jpeg from 'jpeg-js'
import RNFetchBlob from 'rn-fetch-blob'
import ImagePicker from 'react-native-image-picker'
import ImageResizer from 'react-native-image-resizer';

import modelJson from './assets/model/model.json'
import modelWeights from './assets/model/group1-shard1of1.bin'

type Props = {};

type State = {
    isTfReady: boolean;
    isModelReady: boolean;
};

class App extends PureComponent<Props, State> {
    private model: tf.LayersModel | null = null;

    state = {
        isTfReady: false,
        isModelReady: false,
        predictions: null,
        image: null
    };

    async componentDidMount() {
        await tf.ready();
        this.setState({
            isTfReady: true
        });
        try {
            this.model = await tf.loadLayersModel(bundleResourceIO(modelJson, modelWeights), { strict: false });
        } catch (e) {
            console.log(1, e)
        }
        this.setState({ isModelReady: true });

        //////////////////////////////////

        const image = require('./assets/1.jpg');
        console.log(1)
        const imageAssetPath = Image.resolveAssetSource(image);
        console.log(2, imageAssetPath.uri)
        const data = await ImageResizer.createResizedImage('https://realt.by/uploads/tx_uedbhouses/Belarus/5102/1301116904/36/eed358d58c54a818.jpg', 64, 64, 'JPEG', 1);
        console.log(9, data);
        const response = await fetch(imageAssetPath.uri, {}, { isBinary: true });
        console.log(3, response)
        const imageData = await response.arrayBuffer();
        const imageTensor2 = this.imageToTensor(imageData)
        console.log(5, imageTensor2)
        console.log(6, imageTensor2.reshape([1, 797, 1200, 3]));
        console.log(4, imageData)

        // const imageTensor = decodeJpeg(imageData);

        const prediction = await this.model.predict(imageTensor2.reshape([1, 797, 1200, 3]));

// Use prediction in app.
        this.setState({
            prediction,
        });
    }

    imageToTensor(rawImageData) {
        const TO_UINT8ARRAY = true
        const { width, height, data } = jpeg.decode(rawImageData, TO_UINT8ARRAY);
        // Drop the alpha channel info for mobilenet
        const buffer = new Uint8Array(width * height * 3);
        let offset = 0 // offset into original data
        for (let i = 0; i < buffer.length; i += 3) {
            buffer[i] = data[offset];
            buffer[i + 1] = data[offset + 1];
            buffer[i + 2] = data[offset + 2];

            offset += 4
        }

        return tf.tensor3d(buffer, [height, width, 3])
    }

    classifyImage = async () => {
        try {
            const imageAssetPath = Image.resolveAssetSource(this.state.image)
            const response = await fetch(imageAssetPath.uri, {}, { isBinary: true })
            const rawImageData = await response.arrayBuffer()
            const imageTensor = this.imageToTensor(rawImageData)
            const predictions = await this.model.classify(imageTensor)
            this.setState({ predictions })
            console.log(predictions)
        } catch (error) {
            console.log(error)
        }
    }

    selectImage = async () => {
        ImagePicker.launchCamera({ noData: true, maxHeight: 64, maxWidth: 64 }, async (response) => {
            console.log(response);
            try {
                // add cropping
                const data = await ImageResizer.createResizedImage(response.path, 64, 64, 'JPEG', 100);
                console.log(1, data);
                const path = response.uri;
                const response2 = await RNFetchBlob.fs.readFile(data.path, 'ascii');
                //const response = await fetch(`file://${data.path}`, {}, { isBinary: true });
                const imageData = response2;
                const imageTensor2 = this.imageToTensor(imageData);
                console.log(1, imageTensor2)
                console.log(3, tf.tensor([1, 2, 3, 4]).print());
                console.log(33, imageTensor2.div(255).print());
                const prediction = await this.model.predict(imageTensor2.reshape([1, 64, 64, 3]));
                Object.keys(prediction[0]).forEach((el) => console.log(el));
                prediction[0].print();
                prediction[1].print();
                prediction[2].print();
                prediction[3].print();
                prediction[4].print();
                console.log(4, prediction)
            } catch (e) {
                console.log(2, e);
            }
            if (!response.cancelled) {
                const source = { uri: response.uri }
                this.setState({ image: source })
                this.classifyImage()
            }
        });
    }

    renderPrediction = prediction => {
        return (
            <Text key={prediction.className} style={styles.text}>
                {prediction.className}
            </Text>
        )
    }

    render() {
        const { isTfReady, isModelReady, predictions, image } = this.state

        return (
            <View style={styles.container}>
                <StatusBar barStyle='light-content' />
                <View style={styles.loadingContainer}>
                    <Text style={styles.text}>
                        TFJS ready? {isTfReady ? <Text>✅</Text> : ''}
                    </Text>

                    <View style={styles.loadingModelContainer}>
                        <Text style={styles.text}>Model ready? </Text>
                        {isModelReady ? (
                            <Text style={styles.text}>✅</Text>
                        ) : (
                            <ActivityIndicator size='small' />
                        )}
                    </View>
                </View>
                <TouchableOpacity
                    style={styles.imageWrapper}
                    onPress={isModelReady ? this.selectImage : undefined}>
                    {image && <Image source={image} style={styles.imageContainer} />}

                    {isModelReady && !image && (
                        <Text style={styles.transparentText}>Tap to choose image</Text>
                    )}
                </TouchableOpacity>
                <View style={styles.predictionWrapper}>
                    {isModelReady && image && (
                        <Text style={styles.text}>
                            Predictions: {predictions ? '' : 'Predicting...'}
                        </Text>
                    )}
                    {isModelReady &&
                    predictions &&
                    predictions.map(p => this.renderPrediction(p))}
                </View>
            </View>
        )
    }
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#171f24',
        alignItems: 'center'
    },
    loadingContainer: {
        marginTop: 80,
        justifyContent: 'center'
    },
    text: {
        color: '#ffffff',
        fontSize: 16
    },
    loadingModelContainer: {
        flexDirection: 'row',
        marginTop: 10
    },
    imageWrapper: {
        width: 280,
        height: 280,
        padding: 10,
        borderColor: '#cf667f',
        borderWidth: 5,
        borderStyle: 'dashed',
        marginTop: 40,
        marginBottom: 10,
        position: 'relative',
        justifyContent: 'center',
        alignItems: 'center'
    },
    imageContainer: {
        width: 250,
        height: 250,
        position: 'absolute',
        top: 10,
        left: 10,
        bottom: 10,
        right: 10
    },
    predictionWrapper: {
        height: 100,
        width: '100%',
        flexDirection: 'column',
        alignItems: 'center'
    },
    transparentText: {
        color: '#ffffff',
        opacity: 0.7
    },
    footer: {
        marginTop: 40
    },
    poweredBy: {
        fontSize: 20,
        color: '#e69e34',
        marginBottom: 6
    },
    tfLogo: {
        width: 125,
        height: 70
    }
})

export default App