import os
import model
import tensorflow as tf
import argparse


def check_and_makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def convert_str_to_bool(text):
    if text.lower() in ["true", "yes", "y", "1"]:
        return True
    else:
        return False


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--display_step', type=int, default=100)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--gpu', type=int, default=None, choices=[None, 0, 1])

    # Training Parameters
    parser.add_argument('--load_teacher_from_checkpoint', type=str, default="false")
    parser.add_argument('--load_teacher_checkpoint_dir', type=str, default=None)
    parser.add_argument('--model_type', type=str, default="teacher", choices=["teacher", "student"])
    parser.add_argument('--num_steps', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--temperature', type=float, default=5.0)
    parser.add_argument('--dropoutprob', type=float, default=0.75)

    return parser


def setup(args):
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.gpu

    args.load_teacher_from_checkpoint = convert_str_to_bool(args.load_teacher_from_checkpoint)

    check_and_makedir(args.checkpoint_dir)


def main():
    parser = get_parser()
    args = parser.parse_args()
    setup(args)

    tf.reset_default_graph()
    sess = tf.Session()
    if args.model_type == 'student':
        teacher_model = None
        student_model = None
        if args.load_teacher_from_checkpoint:
            teacher_model = model.TeacherModel(sess, num_steps=args.num_steps, dropout_prob=args.dropoutprob,
                                               batch_size=128, learning_rate=0.001, temperature=args.temperature)
            flag = True
            teacher_model.load_model_from_file()
            print("Using Teacher State-------------!")

            stud_sess = tf.Session()
            student_model = model.StudentModel(stud_sess, num_steps=args.num_steps, batch_size=128, learning_rate=0.001,
                                               temperature=args.temperature)
            student_model.train(teacher_model)

        else:
            print("Not Using Teacher State---------!")
            stud_sess = tf.Session()
            student_model = model.StudentModel(stud_sess, num_steps=args.num_steps, batch_size=128, learning_rate=0.001,
                                               temperature=1.0, checkpoint_file='Student_raw')
            student_model.train(teacher_model)

    else:
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        teacher_model = model.TeacherModel(sess, num_steps=args.num_steps, dropout_prob=args.dropoutprob,
                                           batch_size=128, learning_rate=0.001,
                                           temperature=args.temperature)
        teacher_model.train()


if __name__ == '__main__':
    main()
