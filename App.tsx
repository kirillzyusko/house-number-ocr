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
import { bundleResourceIO } from '@tensorflow/tfjs-react-native'
import * as jpeg from 'jpeg-js'
import RNFetchBlob from 'rn-fetch-blob'
import ImagePicker from 'react-native-image-picker'
import ImageResizer from 'react-native-image-resizer';
import ImageCropper from 'react-native-image-crop-picker';
import { Tensor } from "@tensorflow/tfjs-core";

import modelJson from './assets/model/model.json'
import modelWeights from './assets/model/group1-shard1of1.bin'

type Props = {};

type State = {
    isTfReady: boolean;
    isModelReady: boolean;
    predictions: string;
    image: {
        uri: string;
    };
};

const mapPredictionToLabel = async (tensors: Tensor[]): Promise<string> => {
    const lengthTensor = await tensors[0].argMax(-1).data();
    const length = Number(lengthTensor.toString());
    let str = '';

    for (let i = 1; i < length + 1; i++) {
        const digit = await tensors[i].argMax(-1).data();
        str += Number(digit.toString())
    }

    return str;
};

class App extends PureComponent<Props, State> {
    private model: tf.LayersModel | null = null;

    public state = {
        isTfReady: false,
        isModelReady: false,
        predictions: null,
        image: null,
    };

    public async componentDidMount(): Promise<void> {
        await tf.ready();
        this.setState({ isTfReady: true });

        try {
            this.model = await tf.loadLayersModel(bundleResourceIO(modelJson, modelWeights));
        } catch (e) {
            console.error(e);
        }

        this.setState({ isModelReady: true });
    }

    public render(): JSX.Element {
        const { isTfReady, isModelReady, predictions, image } = this.state;

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
                        <Text style={styles.transparentText}>Tap to capture image</Text>
                    )}
                </TouchableOpacity>
                <View style={styles.predictionWrapper}>
                    {isModelReady && image && (
                        <Text style={styles.text}>
                            Predictions: {predictions}
                        </Text>
                    )}
                </View>
            </View>
        )
    }

    private makePrediction = async (response) => {
        this.setState({ predictions: 'Processing...' });

        const croppedImage =
            await ImageCropper.openCropper({
                path: `file://${response.path}`,
                cropping: true,
            });
        const imageData = await RNFetchBlob.fs.readFile(croppedImage.path, 'ascii');
        const tensorTic = new Date().getTime();
        const tensor =
            tf.image.resizeBilinear(this.imageToTensor(imageData), [64,64])
                .reshape([1, 64, 64, 3])
                .div(255);
        const tensorToc = new Date().getTime();
        const tic = new Date().getTime();
        const prediction = await this.model.predict(tensor) as Tensor[];
        const label = await mapPredictionToLabel(prediction);
        const toc = new Date().getTime();

        this.setState({
            predictions: `${label}.\nFeedforward time: ${toc-tic}ms.\nImage resizing time: ${tensorToc-tensorTic}\nTotal time: ${toc-tensorTic}`,
        });
    };

    private imageToTensor = (rawImageData) => {
        const TO_UINT8ARRAY = true;
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
    };

    private selectImage = async () => {
        ImagePicker.launchCamera({ noData: true }, async (response) => {
            if (!response.didCancel) {
                const source = { uri: response.uri };
                this.setState({ image: source });
                await this.makePrediction(response)
            }
        });
    };
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
        fontSize: 16,
        textAlign: 'center',
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
});

export default App