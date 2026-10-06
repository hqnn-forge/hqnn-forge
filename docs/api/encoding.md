# Encoding layers

::: hqnn_forge.encoding

## Input checks

Not exported from `hqnn_forge.encoding`, but used by every encoder to
validate its options and inputs; import them from their module when writing a
new one (see the contributor docs on extending the library).

::: hqnn_forge.encoding.angle_embedding.validate_circuit_options

::: hqnn_forge.encoding.angle_embedding.check_inputs

## Devices and weight shapes

Also not exported from `hqnn_forge.encoding`: the device a layer is built on,
with its fallback chain, and the shape of the variational weights for each
entangler.

::: hqnn_forge.encoding.angle_embedding.resolve_device

::: hqnn_forge.encoding.angle_embedding.is_out_of_memory

::: hqnn_forge.encoding.angle_embedding.KNOWN_DEVICES

::: hqnn_forge.encoding.angle_embedding.FALLBACK_CHAIN

::: hqnn_forge.encoding.angle_embedding.reset_device_fallback

::: hqnn_forge.encoding.angle_embedding.variational_weight_shape
