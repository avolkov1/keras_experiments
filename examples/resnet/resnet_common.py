'''
'''
import os
from collections import defaultdict
from tensorflow.python.ops import data_flow_ops
import tensorflow as tf


__all__ = ('RecordInputImagenetPreprocessor', 'get_num_records',)


def get_num_records(tf_record_pattern):
    def count_records(tf_record_filename):
        count = 0
        for _ in tf.python_io.tf_record_iterator(tf_record_filename):
            count += 1
        return count
    filenames = sorted(tf.gfile.Glob(tf_record_pattern))
    nfile = len(filenames)
    return (count_records(filenames[0]) * (nfile - 1) +
            count_records(filenames[-1]))


class RecordInputImagenetPreprocessor(object):
    '''Preprocessor for images with RecordInput format.'''

    @classmethod
    def _deserialize_image_record(cls, record):
        feature_map = {
            'image/encoded': tf.FixedLenFeature([], tf.string, ''),
            'image/class/label': tf.FixedLenFeature([1], tf.int64, -1),
            'image/class/text': tf.FixedLenFeature([], tf.string, ''),
            'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32)
        }
        with tf.name_scope('deserialize_image_record'):
            obj = tf.parse_single_example(record, feature_map)
            imgdata = obj['image/encoded']
            label = tf.cast(obj['image/class/label'], tf.int32)
            bbox = tf.stack([obj['image/object/bbox/%s' % x].values
                             for x in ['ymin', 'xmin', 'ymax', 'xmax']])
            bbox = tf.transpose(tf.expand_dims(bbox, 0), [0, 2, 1])
            text = obj['image/class/text']
            return imgdata, label, bbox, text

    @classmethod
    def _decode_jpeg(cls, imgdata, channels=3):
        return tf.image.decode_jpeg(imgdata, channels=channels,
                                    fancy_upscaling=False,
                                    dct_method='INTEGER_FAST')

    @classmethod
    def _decode_png(cls, imgdata, channels=3):
        return tf.image.decode_png(imgdata, channels=channels)

    @classmethod
    def _random_crop_and_resize_image(cls, image, bbox, height, width,
                                      val=False):
        with tf.name_scope('random_crop_and_resize'):
            if not val:
                # bbox_begin, bbox_size, distorted_bbox = \
                bbox_begin, bbox_size, _ = \
                    tf.image.sample_distorted_bounding_box(
                        tf.shape(image),
                        bounding_boxes=bbox,
                        min_object_covered=0.1,
                        aspect_ratio_range=[0.8, 1.25],
                        area_range=[0.1, 1.0],
                        max_attempts=100,
                        use_image_if_no_bounding_boxes=True)
                # Crop the image to the distorted bounding box
                image = tf.slice(image, bbox_begin, bbox_size)
            # Resize to the desired output size
            image = tf.image.resize_images(
                image,
                [height, width],
                tf.image.ResizeMethod.BILINEAR,
                align_corners=False)
            image.set_shape([height, width, 3])
            return image

    @classmethod
    def distort_image_color(cls, image, order):
        with tf.name_scope('distort_color'):
            image /= 255.

            def brightness(img):
                return tf.image.random_brightness(img, max_delta=32. / 255.)

            def saturation(img):
                return tf.image.random_saturation(img, lower=0.5, upper=1.5)

            def hue(img):
                return tf.image.random_hue(img, max_delta=0.2)

            def contrast(img):
                return tf.image.random_contrast(img, lower=0.5, upper=1.5)

            if order == 0:
                ops = [brightness, saturation, hue, contrast]
            else:
                ops = [brightness, contrast, saturation, hue]
            for op in ops:
                image = op(image)
            # The random_* ops do not necessarily clamp the output range
            image = tf.clip_by_value(image, 0.0, 1.0)
            # Restore the original scaling
            image *= 255
            return image

    @classmethod
    def _preprocess(cls, imgdata, bbox, thread_id, height, width,
                    distort_color, val=False):
        with tf.name_scope('preprocess_image'):
            try:
                image = cls._decode_jpeg(imgdata)
            except Exception:
                image = cls._decode_png(imgdata)
    #         if thread_id < self.nsummary:
    #             image_with_bbox = tf.image.draw_bounding_boxes(
    #                 tf.expand_dims(tf.to_float(image), 0), bbox)
    #             tf.summary.image('original_image_and_bbox', image_with_bbox)
            image = cls._random_crop_and_resize_image(
                image, bbox, height, width, val)
    #         if thread_id < self.nsummary:
    #             tf.summary.image('cropped_resized_image',
    #                              tf.expand_dims(image, 0))
            if not val:
                image = tf.image.random_flip_left_right(image)
    #         if thread_id < self.nsummary:
    #             tf.summary.image('flipped_image',
    #                              tf.expand_dims(image, 0))
            if distort_color and not val:
                image = cls.distort_image_color(image, order=thread_id % 2)
    #             if thread_id < self.nsummary:
    #                 tf.summary.image('distorted_color_image',
    #                                  tf.expand_dims(image, 0))
        return image

    @classmethod
    def device_minibatches(cls, num_devices, data_dir, total_batch_size,
                           height, width, distort_color,
                           val=False):
        dtype = tf.float32
        subset = 'validation' if val else 'train'

        nrecord = get_num_records(os.path.join(
            data_dir, '{}-*'.format(subset)))
        input_buffer_size = min(10000, nrecord)

        record_input = data_flow_ops.RecordInput(
            file_pattern=os.path.join(data_dir, '{}-*'.format(subset)),
            parallelism=64,
            # Note: This causes deadlock during init if
            # larger than dataset
            buffer_size=input_buffer_size,
            batch_size=total_batch_size,
            seed=0)

        records = record_input.get_yield_op()

        # Split batch into individual images
        records = tf.split(records, total_batch_size, 0)
        records = [tf.reshape(record, []) for record in records]
        # Deserialize and preprocess images into batches for each device
        images = defaultdict(list)
        labels = defaultdict(list)
        with tf.name_scope('input_pipeline'):
            for thread_id, record in enumerate(records):
                imgdata, label, bbox, _ = cls._deserialize_image_record(record)
                image = cls._preprocess(
                    imgdata, bbox, thread_id, height, width, distort_color,
                    val=val)
                label -= 1  # Change to 0-based (don't use background class)
                device_num = thread_id % num_devices
                images[device_num].append(image)
                labels[device_num].append(label)

            # Stack images back into a sub-batch for each device
            for device_num in xrange(num_devices):
                images[device_num] = tf.parallel_stack(images[device_num])
                labels[device_num] = tf.concat(labels[device_num], 0)
                images[device_num] = tf.reshape(
                    images[device_num], [-1, height, width, 3])
                images[device_num] = tf.clip_by_value(
                    images[device_num], 0., 255.)
                images[device_num] = tf.cast(images[device_num], dtype)

        return images, labels, nrecord
