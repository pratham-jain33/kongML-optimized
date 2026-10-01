"""Note-event post-processing for Kong's piano transcription model.

Vendored (numpy-only subset) from Qiuqiang Kong's piano_transcription_inference
package (MIT License, Copyright (c) Qiuqiang Kong):
  - RegressionPostProcessor class from utilities.py
  - note_detection_with_onset_offset_regress / pedal_detection_with_onset_offset_regress
    from piano_vad.py
  - constants from config.py

Vendored so the transcribe service does not need the full package (whose
__init__ imports torch, ~400MB RAM). Behavior is identical to upstream.
"""
import numpy as np

# --- config.py constants ---
SAMPLE_RATE = 16000
CLASSES_NUM = 88
BEGIN_NOTE = 21  # MIDI note of A0
FRAMES_PER_SECOND = 100
VELOCITY_SCALE = 128


# --- piano_vad.py ---
def note_detection_with_onset_offset_regress(frame_output, onset_output,
    onset_shift_output, offset_output, offset_shift_output, velocity_output,
    frame_threshold):
    """Process prediction matrices to note events information.
    First, detect onsets with onset outputs. Then, detect offsets
    with frame and offset outputs.

    Args:
      frame_output: (frames_num,)
      onset_output: (frames_num,)
      onset_shift_output: (frames_num,)
      offset_output: (frames_num,)
      offset_shift_output: (frames_num,)
      velocity_output: (frames_num,)
      frame_threshold: float
    Returns:
      output_tuples: list of [bgn, fin, onset_shift, offset_shift, normalized_velocity]
    """
    output_tuples = []
    bgn = None
    frame_disappear = None
    offset_occur = None

    for i in range(onset_output.shape[0]):
        if onset_output[i] == 1:
            """Onset detected"""
            if bgn:
                """Consecutive onsets. E.g., pedal is not released, but two
                consecutive notes being played."""
                fin = max(i - 1, 0)
                output_tuples.append([bgn, fin, onset_shift_output[bgn],
                    0, velocity_output[bgn]])
                frame_disappear, offset_occur = None, None
            bgn = i

        if bgn and i > bgn:
            """If onset found, then search offset"""
            if frame_output[i] <= frame_threshold and not frame_disappear:
                """Frame disappear detected"""
                frame_disappear = i

            if offset_output[i] == 1 and not offset_occur:
                """Offset detected"""
                offset_occur = i

            if frame_disappear:
                if offset_occur and offset_occur - bgn > frame_disappear - offset_occur:
                    """bgn --------- offset_occur --- frame_disappear"""
                    fin = offset_occur
                else:
                    """bgn --- offset_occur --------- frame_disappear"""
                    fin = frame_disappear
                output_tuples.append([bgn, fin, onset_shift_output[bgn],
                    offset_shift_output[fin], velocity_output[bgn]])
                bgn, frame_disappear, offset_occur = None, None, None

            if bgn and (i - bgn >= 600 or i == onset_output.shape[0] - 1):
                """Offset not detected"""
                fin = i
                output_tuples.append([bgn, fin, onset_shift_output[bgn],
                    offset_shift_output[fin], velocity_output[bgn]])
                bgn, frame_disappear, offset_occur = None, None, None

    # Sort pairs by onsets
    output_tuples.sort(key=lambda pair: pair[0])

    return output_tuples


def pedal_detection_with_onset_offset_regress(frame_output, offset_output,
    offset_shift_output, frame_threshold):
    """Process prediction array to pedal events information."""
    output_tuples = []
    bgn = None
    frame_disappear = None
    offset_occur = None

    for i in range(1, frame_output.shape[0]):
        if frame_output[i] >= frame_threshold and frame_output[i] > frame_output[i - 1]:
            """Pedal onset detected"""
            if bgn:
                pass
            else:
                bgn = i

        if bgn and i > bgn:
            """If onset found, then search offset"""
            if frame_output[i] <= frame_threshold and not frame_disappear:
                """Frame disappear detected"""
                frame_disappear = i

            if offset_output[i] == 1 and not offset_occur:
                """Offset detected"""
                offset_occur = i

            if offset_occur:
                fin = offset_occur
                output_tuples.append([bgn, fin, 0., offset_shift_output[fin]])
                bgn, frame_disappear, offset_occur = None, None, None

            if frame_disappear and i - frame_disappear >= 10:
                """offset not detected but frame disappear"""
                fin = frame_disappear
                output_tuples.append([bgn, fin, 0., offset_shift_output[fin]])
                bgn, frame_disappear, offset_occur = None, None, None

    # Sort pairs by onsets
    output_tuples.sort(key=lambda pair: pair[0])

    return output_tuples


# --- utilities.py: RegressionPostProcessor ---
class RegressionPostProcessor(object):
    def __init__(self, frames_per_second, classes_num, onset_threshold,
        offset_threshold, frame_threshold, pedal_offset_threshold):
        """Postprocess the output probabilities of a transcription model to MIDI
        events.

        Args:
          frames_per_second: int
          classes_num: int
          onset_threshold: float
          offset_threshold: float
          frame_threshold: float
          pedal_offset_threshold: float
        """
        self.frames_per_second = frames_per_second
        self.classes_num = classes_num
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.frame_threshold = frame_threshold
        self.pedal_offset_threshold = pedal_offset_threshold
        self.begin_note = BEGIN_NOTE
        self.velocity_scale = VELOCITY_SCALE

    def output_dict_to_midi_events(self, output_dict):
        """Main function. Post process model outputs to MIDI events."""
        (est_on_off_note_vels, est_pedal_on_offs) = \
            self.output_dict_to_note_pedal_arrays(output_dict)

        est_note_events = self.detected_notes_to_events(est_on_off_note_vels)

        if est_pedal_on_offs is None:
            est_pedal_events = None
        else:
            est_pedal_events = self.detected_pedals_to_events(est_pedal_on_offs)

        return est_note_events, est_pedal_events

    def output_dict_to_note_pedal_arrays(self, output_dict):
        """Postprocess the output probabilities of a transcription model to MIDI
        events. Returns (est_on_off_note_vels, est_pedal_on_offs)."""
        # ------ 1. Process regression outputs to binarized outputs ------
        (onset_output, onset_shift_output) = \
            self.get_binarized_output_from_regression(
                reg_output=output_dict['reg_onset_output'],
                threshold=self.onset_threshold, neighbour=2)

        output_dict['onset_output'] = onset_output
        output_dict['onset_shift_output'] = onset_shift_output

        (offset_output, offset_shift_output) = \
            self.get_binarized_output_from_regression(
                reg_output=output_dict['reg_offset_output'],
                threshold=self.offset_threshold, neighbour=4)

        output_dict['offset_output'] = offset_output
        output_dict['offset_shift_output'] = offset_shift_output

        if 'reg_pedal_offset_output' in output_dict.keys():
            (pedal_offset_output, pedal_offset_shift_output) = \
                self.get_binarized_output_from_regression(
                    reg_output=output_dict['reg_pedal_offset_output'],
                    threshold=self.pedal_offset_threshold, neighbour=4)

            output_dict['pedal_offset_output'] = pedal_offset_output
            output_dict['pedal_offset_shift_output'] = pedal_offset_shift_output

        # ------ 2. Process matrices results to event results ------
        est_on_off_note_vels = self.output_dict_to_detected_notes(output_dict)

        if 'reg_pedal_onset_output' in output_dict.keys():
            est_pedal_on_offs = self.output_dict_to_detected_pedals(output_dict)
        else:
            est_pedal_on_offs = None

        return est_on_off_note_vels, est_pedal_on_offs

    def get_binarized_output_from_regression(self, reg_output, threshold, neighbour):
        """Calculate binarized output and shifts of onsets or offsets from the
        regression results."""
        binary_output = np.zeros_like(reg_output)
        shift_output = np.zeros_like(reg_output)
        (frames_num, classes_num) = reg_output.shape

        for k in range(classes_num):
            x = reg_output[:, k]
            for n in range(neighbour, frames_num - neighbour):
                if x[n] > threshold and self.is_monotonic_neighbour(x, n, neighbour):
                    binary_output[n, k] = 1

                    """See Section III-D in [1] for deduction.
                    [1] Q. Kong, et al., High-resolution Piano Transcription
                    with Pedals by Regressing Onsets and Offsets Times, 2020."""
                    if x[n - 1] > x[n + 1]:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n + 1]) / 2
                    else:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n - 1]) / 2
                    shift_output[n, k] = shift

        return binary_output, shift_output

    def is_monotonic_neighbour(self, x, n, neighbour):
        """Detect if values are monotonic in both side of x[n]."""
        monotonic = True
        for i in range(neighbour):
            if x[n - i] < x[n - i - 1]:
                monotonic = False
            if x[n + i] < x[n + i + 1]:
                monotonic = False

        return monotonic

    def output_dict_to_detected_notes(self, output_dict):
        """Postprocess output_dict to piano notes. Returns (notes, 4) array:
        onset, offset, MIDI note, velocity."""
        est_tuples = []
        est_midi_notes = []
        classes_num = output_dict['frame_output'].shape[-1]

        for piano_note in range(classes_num):
            """Detect piano notes"""
            est_tuples_per_note = note_detection_with_onset_offset_regress(
                frame_output=output_dict['frame_output'][:, piano_note],
                onset_output=output_dict['onset_output'][:, piano_note],
                onset_shift_output=output_dict['onset_shift_output'][:, piano_note],
                offset_output=output_dict['offset_output'][:, piano_note],
                offset_shift_output=output_dict['offset_shift_output'][:, piano_note],
                velocity_output=output_dict['velocity_output'][:, piano_note],
                frame_threshold=self.frame_threshold)

            est_tuples += est_tuples_per_note
            est_midi_notes += [piano_note + self.begin_note] * len(est_tuples_per_note)

        est_tuples = np.array(est_tuples)   # (notes, 5)
        est_midi_notes = np.array(est_midi_notes)  # (notes,)

        if len(est_tuples) == 0:
            return np.array([])

        else:
            onset_times = (est_tuples[:, 0] + est_tuples[:, 2]) / self.frames_per_second
            offset_times = (est_tuples[:, 1] + est_tuples[:, 3]) / self.frames_per_second
            velocities = est_tuples[:, 4]

            est_on_off_note_vels = np.stack(
                (onset_times, offset_times, est_midi_notes, velocities), axis=-1)
            est_on_off_note_vels = est_on_off_note_vels.astype(np.float32)

            return est_on_off_note_vels

    def output_dict_to_detected_pedals(self, output_dict):
        """Postprocess output_dict to piano pedals."""
        est_tuples = pedal_detection_with_onset_offset_regress(
            frame_output=output_dict['pedal_frame_output'][:, 0],
            offset_output=output_dict['pedal_offset_output'][:, 0],
            offset_shift_output=output_dict['pedal_offset_shift_output'][:, 0],
            frame_threshold=0.5)

        est_tuples = np.array(est_tuples)

        if len(est_tuples) == 0:
            return np.array([])

        else:
            onset_times = (est_tuples[:, 0] + est_tuples[:, 2]) / self.frames_per_second
            offset_times = (est_tuples[:, 1] + est_tuples[:, 3]) / self.frames_per_second
            est_on_off = np.stack((onset_times, offset_times), axis=-1)
            est_on_off = est_on_off.astype(np.float32)
            return est_on_off

    def detected_notes_to_events(self, est_on_off_note_vels):
        """Reformat detected notes to midi events."""
        midi_events = []
        for i in range(est_on_off_note_vels.shape[0]):
            midi_events.append({
                'onset_time': est_on_off_note_vels[i][0],
                'offset_time': est_on_off_note_vels[i][1],
                'midi_note': int(est_on_off_note_vels[i][2]),
                'velocity': int(est_on_off_note_vels[i][3] * self.velocity_scale)})

        return midi_events

    def detected_pedals_to_events(self, pedal_on_offs):
        """Reformat detected pedal onset and offsets to events."""
        pedal_events = []
        for i in range(len(pedal_on_offs)):
            pedal_events.append({
                'onset_time': pedal_on_offs[i, 0],
                'offset_time': pedal_on_offs[i, 1]})

        return pedal_events
