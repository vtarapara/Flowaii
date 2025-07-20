from matplotlib.cbook import CallbackRegistry
import tensorflow as tf
import numpy as np

# This file contains the generic MLPF model definitions
# PFNetDense: the GNN-based model with graph building based on LSH and a Gaussian distance kernel

'''
 - make_model returns make_gnn_dense(config, dtype) returns model = PFNetDense()
    - inputs X -> input encoding (ffn) -> graph layer (cgn) -> output decoding (more models)
    Detail
        - assigns inputs to X
        - applies input encoding to get X_enc
        - masks out elements where the third dimension is 0
        - makes a mask object identical to x filtered by zero elements, and adds a dimension at the end ("innermost")
        - if skip connections are needed, X_enc without mask is added to a list.
        - Then, X_enc_cg, input to the combined graph layer, is created.
        - If input must be encoded, which it is for this graphyness:
        - X_enc_ffn is set to the activation(of a single pointwise feed forward network   (tf.lazy.Dense())     
            - number, size of output desired
            - number, size of interior layers
            - "node_encoding" //name of this layer
         - X_enc_cg becomes X_enc_ffn
        - Over the list of CG layers, they are run, and outputs are concatenated.
        - decodor output is the masked concatenation of encoding layers
        - Special energy layers also concatenated
        - Everything sent to output_decoding [X_enc, dec_output, dec_output_energy, msk_input], training=training)

        if self.debug:
            for k in debugging_data.keys():
                ret[k] = debugging_data[k]

        if self.multi_output:
            return ret
        else:
            return tf.concat([ret["cls"], ret["charge"], ret["pt"], ret["eta"], ret["sin_phi"], ret["cos_phi"], ret["energy"]], axis=-1)

 
'''

class InputEncodingATLAS(tf.keras.layers.Layer):
    """Encode ATLAS data.  Expects as input:
        0. category (cast to int32, includes electric charge)
        1. pt
        2. eta
        3. phi
        4. energy
        5. eem 
    Encoded data is:  [typ, pt, sqrt(pt), pt^2, (pz/pt), (p/pt), abseta, sin phi, cos phi ]
    """     
    def __init__(self, num_input_classes):
        """Class to encode ATLAS data.
        @param num_input_classes : int (count of unique TYP in dataset)"""
        super(InputEncodingATLAS, self).__init__()
        self.num_input_classes = num_input_classes

    """
        X: [Nbatch, Nelem, Nfeat] array of all the input detector element feature data
    """        
    @tf.function
    def call(self, X):
        log_energy = tf.expand_dims(tf.math.log(X[:, :, 4]+1.0), axis=-1)

        #X[:, :, 0] - categorical index of the element type
        Xid = tf.cast(tf.one_hot(tf.cast(X[:, :, 0], tf.int32), self.num_input_classes), dtype=X.dtype)
        #Xpt = tf.expand_dims(tf.math.log1p(X[:, :, 1]), axis=-1)
        Xpt = tf.expand_dims(tf.math.log(X[:, :, 1] + 1.0), axis=-1)

        Xpt_0p5 = tf.math.sqrt(Xpt)
        Xpt_2 = tf.math.pow(Xpt, 2)

        Xeta1 = tf.clip_by_value(tf.expand_dims(tf.sinh(X[:, :, 2]), axis=-1), -10, 10)
        Xeta2 = tf.clip_by_value(tf.expand_dims(tf.cosh(X[:, :, 2]), axis=-1), -10, 10)
        Xabs_eta = tf.expand_dims(tf.math.abs(X[:, :, 2]), axis=-1)
        Xphi1 = tf.expand_dims(tf.sin(X[:, :, 3]), axis=-1)
        Xphi2 = tf.expand_dims(tf.cos(X[:, :, 3]), axis=-1)

        #Xe = tf.expand_dims(tf.math.log1p(X[:, :, 4]), axis=-1)
        Xe = log_energy
        Xe_0p5 = tf.math.sqrt(log_energy)
        Xe_2 = tf.math.pow(log_energy, 2)

        Xe_transverse = log_energy - tf.math.log(Xeta2)

        return tf.concat([
            Xid,
            Xpt, Xpt_0p5, Xpt_2,
            Xeta1, Xeta2,
            Xabs_eta,
            Xphi1, Xphi2,
            Xe, Xe_0p5, Xe_2,
            Xe_transverse,
            X], axis=-1
        )

class OutputDecoding(tf.keras.Model):
    def __init__(self,
        activation="elu",
        regression_use_classification=True,
        num_output_classes=8,
        schema="atlas",
        dropout=0.0,

        pt_skip_gate=True,
        eta_skip_gate=True,
        phi_skip_gate=True,
        energy_skip_gate=True,

        id_dim_decrease=True,
        charge_dim_decrease=True,
        pt_dim_decrease=False,
        eta_dim_decrease=False,
        phi_dim_decrease=False,
        energy_dim_decrease=False,

        id_hidden_dim=128,
        charge_hidden_dim=128,
        pt_hidden_dim=128,
        eta_hidden_dim=128,
        phi_hidden_dim=128,
        energy_hidden_dim=128,

        id_num_layers=4,
        charge_num_layers=2,
        pt_num_layers=3,
        eta_num_layers=3,
        phi_num_layers=3,
        energy_num_layers=3,

        layernorm=False,
        mask_reg_cls0=True,
        energy_multimodal=True,
        **kwargs):

        super(OutputDecoding, self).__init__(**kwargs)

        self.regression_use_classification = regression_use_classification
        self.schema = schema
        self.dropout = dropout

        self.pt_skip_gate = pt_skip_gate
        self.eta_skip_gate = eta_skip_gate
        self.phi_skip_gate = phi_skip_gate

        self.mask_reg_cls0 = mask_reg_cls0

        self.energy_multimodal = energy_multimodal

        self.do_layernorm = layernorm
        if self.do_layernorm:
            self.layernorm = tf.keras.layers.LayerNormalization(axis=-1, name="output_layernorm")

        self.ffn_id = point_wise_feed_forward_network(
            num_output_classes, id_hidden_dim,
            "ffn_cls",
            num_layers=id_num_layers,
            activation=activation,
            dim_decrease=id_dim_decrease,
            dropout=dropout
        )
        self.ffn_charge = point_wise_feed_forward_network(
            1, charge_hidden_dim,
            "ffn_charge",
            num_layers=charge_num_layers,
            activation=activation,
            dim_decrease=charge_dim_decrease,
            dropout=dropout
        )
        
        self.ffn_pt = point_wise_feed_forward_network(
            2, pt_hidden_dim, "ffn_pt",
            num_layers=pt_num_layers,
            activation=activation,
            dim_decrease=pt_dim_decrease,
            dropout=dropout
        )

        self.ffn_eta = point_wise_feed_forward_network(
            2, eta_hidden_dim, "ffn_eta",
            num_layers=eta_num_layers,
            activation=activation,
            dim_decrease=eta_dim_decrease,
            dropout=dropout
        )

        self.ffn_phi = point_wise_feed_forward_network(
            4, phi_hidden_dim, "ffn_phi",
            num_layers=phi_num_layers,
            activation=activation,
            dim_decrease=phi_dim_decrease,
            dropout=dropout
        )

        self.ffn_energy = point_wise_feed_forward_network(
            num_output_classes if self.energy_multimodal else 1, energy_hidden_dim, "ffn_energy",
            num_layers=energy_num_layers,
            activation=activation,
            dim_decrease=energy_dim_decrease,
            dropout=dropout)

    """
    X_input: (n_batch, n_elements, n_input_features) raw node input features
    X_encoded: (n_batch, n_elements, n_encoded_features) encoded/transformed node features
    msk_input: (n_batch, n_elements) boolean mask of active nodes
    """
    def call(self, args, training=False):

        X_input, X_encoded, X_encoded_energy, msk_input = args

        if self.do_layernorm:
            X_encoded = self.layernorm(X_encoded)

        out_id_logits = self.ffn_id(X_encoded, training=training)*msk_input

        out_id_softmax = tf.nn.softmax(out_id_logits, axis=-1)
        out_id_hard_softmax = tf.stop_gradient(tf.nn.softmax(100*out_id_logits, axis=-1))
        out_charge = self.ffn_charge(X_encoded, training=training)*msk_input

        orig_eta = X_input[:, :, 2:3]

        #FIXME: better schema propagation 
        #skip connection from raw input values
        if self.schema == "atlas":
            orig_sin_phi = tf.math.sin(X_input[:, :, 3:4])*msk_input
            orig_cos_phi = tf.math.cos(X_input[:, :, 3:4])*msk_input
            orig_energy = X_input[:, :, 4:5]*msk_input
        elif self.schema == "delphes":
            orig_sin_phi = X_input[:, :, 3:4]*msk_input
            orig_cos_phi = X_input[:, :, 4:5]*msk_input
            orig_energy = X_input[:, :, 5:6]*msk_input

        if self.regression_use_classification:
            X_encoded = tf.concat([X_encoded, tf.stop_gradient(out_id_logits)], axis=-1)

        pred_eta_corr = self.ffn_eta(X_encoded, training=training)*msk_input
        pred_phi_corr = self.ffn_phi(X_encoded, training=training)*msk_input

        if self.eta_skip_gate:
            eta_gate = tf.keras.activations.sigmoid(pred_eta_corr[:, :, 0:1])
            pred_eta = orig_eta + pred_eta_corr[:, :, 1:2]
        else:
            pred_eta = orig_eta*pred_eta_corr[:, :, 0:1] + pred_eta_corr[:, :, 1:2]
        
        if self.phi_skip_gate:
            sin_phi_gate = tf.keras.activations.sigmoid(pred_phi_corr[:, :, 0:1])
            cos_phi_gate = tf.keras.activations.sigmoid(pred_phi_corr[:, :, 2:3])
            pred_sin_phi = orig_sin_phi + pred_phi_corr[:, :, 1:2]
            pred_cos_phi = orig_cos_phi + pred_phi_corr[:, :, 3:4]
        else:
            pred_sin_phi = orig_sin_phi*pred_phi_corr[:, :, 0:1] + pred_phi_corr[:, :, 1:2]
            pred_cos_phi = orig_cos_phi*pred_phi_corr[:, :, 2:3] + pred_phi_corr[:, :, 3:4]

        X_encoded_energy = tf.concat([X_encoded, X_encoded_energy], axis=-1)
        if self.regression_use_classification:
            X_encoded_energy = tf.concat([X_encoded_energy, tf.stop_gradient(out_id_logits)], axis=-1)

        pred_energy_corr = self.ffn_energy(X_encoded_energy, training=training)*msk_input

        #In case of a multimodal prediction, weight the per-class energy predictions by the approximately one-hot vector
        if self.energy_multimodal:
            pred_energy = tf.reduce_sum(out_id_hard_softmax*pred_energy_corr, axis=-1, keepdims=True)
        else:
            pred_energy = pred_energy_corr

        #compute pt=E/cosh(eta)
        orig_pt = tf.stop_gradient(pred_energy/tf.math.cosh(tf.clip_by_value(pred_eta, -8, 8)))

        pred_pt_corr = self.ffn_pt(X_encoded_energy, training=training)*msk_input
        if self.pt_skip_gate:
            pt_gate = tf.keras.activations.sigmoid(pred_pt_corr[:, :, 0:1])
            pred_pt = orig_pt + pt_gate*pred_pt_corr[:, :, 1:2]
        else:
            pred_pt = orig_pt*pred_pt_corr[:, :, 0:1] + pred_pt_corr[:, :, 1:2]
        
        #mask the regression outputs for the nodes with a class prediction 0
        msk_output = tf.expand_dims(tf.cast(tf.argmax(out_id_hard_softmax, axis=-1)!=0, tf.float32), axis=-1)

        if self.mask_reg_cls0:
            out_charge = out_charge*msk_output
            pred_pt = pred_pt*msk_output
            pred_eta = pred_eta*msk_output
            pred_sin_phi = pred_sin_phi*msk_output
            pred_cos_phi = pred_cos_phi*msk_output
            pred_energy = pred_energy*msk_output

        ret = {
            "cls": out_id_softmax,
            "charge": out_charge*msk_input,
            "pt": pred_pt*msk_input,
            "eta": pred_eta*msk_input,
            "sin_phi": pred_sin_phi*msk_input,
            "cos_phi": pred_cos_phi*msk_input,
            "energy": pred_energy*msk_input,

            #per-event sum of energy and pt
            "sum_energy": tf.reduce_sum(pred_energy*msk_input*msk_output, axis=-2),
            "sum_pt": tf.reduce_sum(pred_pt*msk_input*msk_output, axis=-2),
        }

        return ret

    def set_trainable_regression(self):
        self.ffn_id.trainable = False
        self.ffn_charge.trainable = False
        self.ffn_phi.trainable = False
        self.ffn_eta.trainable = False
        self.ffn_pt.trainable = False
        self.ffn_energy.trainable = True

    def set_trainable_classification(self):
        self.ffn_id.trainable = True
        self.ffn_charge.trainable = True
        self.ffn_phi.trainable = False
        self.ffn_eta.trainable = False
        self.ffn_pt.trainable = False
        self.ffn_energy.trainable = False


regularizer_weight = 0.0
def point_wise_feed_forward_network(d_model,  dff,     name,     num_layers=1, activation='elu', dtype=tf.dtypes.float32, dim_decrease=False, dropout=0.0):
    """ A stack of regular dense layers.
        @param d_model : dimensionality of the output space (units)
        @param dff : dimension of the feed-forward layer output?
    """    

    if regularizer_weight > 0:
        bias_regularizer =  tf.keras.regularizers.L1(regularizer_weight)
        kernel_regularizer = tf.keras.regularizers.L1(regularizer_weight)
    else:
        bias_regularizer = None
        kernel_regularizer = None

    layers = []
    for ilayer in range(num_layers):
        _name = name + "_dense_{}".format(ilayer)

        layers.append(tf.keras.layers.Dense(
            dff, activation=activation, bias_regularizer=bias_regularizer,
            kernel_regularizer=kernel_regularizer, name=_name))

        if dropout>0.0: 
            layers.append(tf.keras.layers.Dropout(dropout)) #applied to the layer just before

        if dim_decrease:
            dff = dff // 2 #floor division

    layers.append(tf.keras.layers.Dense(d_model, dtype=dtype, name="{}_dense_{}".format(name, ilayer+1)))
    return tf.keras.Sequential(layers, name=name) #now stack these layers


class PFNetDense(tf.keras.Model):
    '''MLPF PFNetDense cloned here for debugging'''
    def __init__(self,
            do_node_encoding=False,
            node_encoding_hidden_dim=128,
            dropout=0.0,
            activation="gelu",
            multi_output=False,
            num_input_classes=8,
            num_output_classes=3,
            num_graph_layers_common=1,
            num_graph_layers_energy=1,
            input_encoding="atlas",
            skip_connection=True,
            graph_kernel={},
            combined_graph_layer={},
            node_message={},
            output_decoding={},
            debug=False,
            schema="cms",
            node_update_mode="concat",
            **kwargs
        ):
        super(PFNetDense, self).__init__() #initialize the base class

        self.multi_output = multi_output
        self.debug = debug

        self.skip_connection = skip_connection
        
        self.do_node_encoding = do_node_encoding
        self.node_encoding_hidden_dim = node_encoding_hidden_dim
        self.dropout = dropout
        self.node_update_mode = node_update_mode
        self.activation = getattr(tf.keras.activations, activation)

        if self.do_node_encoding:
            self.node_encoding = point_wise_feed_forward_network(
                combined_graph_layer["node_message"]["output_dim"],
                self.node_encoding_hidden_dim,
                "node_encoding",
                num_layers=1,
                activation=self.activation,
                dropout=self.dropout
            )

        if input_encoding == "atlas":
            self.enc = InputEncodingATLAS(num_input_classes)
        elif input_encoding == "default":
            self.enc = InputEncoding(num_input_classes)

        self.cg = [CombinedGraphLayer(name="cg_{}".format(i), **combined_graph_layer) for i in range(num_graph_layers_common)]
        self.cg_energy = [CombinedGraphLayer(name="cg_energy_{}".format(i), **combined_graph_layer) for i in range(num_graph_layers_energy)]

        output_decoding["schema"] = schema
        output_decoding["num_output_classes"] = num_output_classes
        self.output_dec = OutputDecoding(**output_decoding)

    def call(self, inputs, training=False):
        X = inputs
        debugging_data = {}

        #encode the elements for classification (id)
        X_enc = self.enc(X)

        #mask padded elements
        msk = X[:, :, 0] != 0
        msk_input = tf.expand_dims(tf.cast(msk, X_enc.dtype), -1)

        encs = []
        if self.skip_connection:
            encs.append(X_enc)

        X_enc_cg = X_enc
        if self.do_node_encoding:
            X_enc_ffn = self.activation(self.node_encoding(X_enc_cg, training=training))
            X_enc_cg = X_enc_ffn

        for cg in self.cg:
            enc_all = cg(X_enc_cg, msk, training=training)

            if self.node_update_mode == "additive":
                X_enc_cg += enc_all["enc"]
            elif self.node_update_mode == "concat":
                X_enc_cg = enc_all["enc"]
                encs.append(X_enc_cg)

            if self.debug:
                debugging_data[cg.name] = enc_all
        
        if self.node_update_mode == "concat":
            dec_output = tf.concat(encs, axis=-1)*msk_input
        elif self.node_update_mode == "additive":
            dec_output = X_enc_cg

        X_enc_cg = X_enc
        if self.do_node_encoding:
            X_enc_cg = X_enc_ffn

        encs_energy = []
        for cg in self.cg_energy:
            enc_all = cg(X_enc_cg, msk, training=training)
            if self.node_update_mode == "additive":
                X_enc_cg += enc_all["enc"]
            elif self.node_update_mode == "concat":
                X_enc_cg = enc_all["enc"]
                encs_energy.append(X_enc_cg)

            if self.debug:
                debugging_data[cg.name] = enc_all
            encs_energy.append(X_enc_cg)

        if self.node_update_mode == "concat":
            dec_output_energy = tf.concat(encs_energy, axis=-1)*msk_input
        elif self.node_update_mode == "additive":
            dec_output_energy = X_enc_cg

        if self.debug:
            debugging_data["dec_output"] = dec_output
            debugging_data["dec_output_energy"] = dec_output_energy

        ret = self.output_dec([X_enc, dec_output, dec_output_energy, msk_input], training=training)

        if self.debug:
            for k in debugging_data.keys():
                ret[k] = debugging_data[k]

        if self.multi_output:
            return ret
        else:
            return tf.concat([ret["cls"], ret["charge"], ret["pt"], ret["eta"], ret["sin_phi"], ret["cos_phi"], ret["energy"]], axis=-1)

    def set_trainable_named(self, layer_names):
        self.trainable = True

        for layer in self.layers:
            layer.trainable = False

        self.output_dec.set_trainable_named(layer_names)

    