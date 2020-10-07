from model import GIKT
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
from data_process import DataGenerator

def train(args,train_dkt):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:

        print(args.model)

        model = GIKT(args)

        saver = tf.train.Saver()
        index = 0

        if train_dkt:
            # lr = 0.4
            # lr_decay = 0.92
            sess.run(tf.global_variables_initializer())

            model_dir = save_model_dir(args)

            best_valid_auc = 0

            for epoch in tqdm(range(args.num_epochs)):
                train_generator = DataGenerator(args.train_seqs, args.max_step, batch_size=args.batch_size,
                                                feature_size=args.feature_answer_size - 2,
                                                hist_num=args.hist_neighbor_num)
                valid_generator = DataGenerator(args.valid_seqs, args.max_step, batch_size=args.batch_size,
                                                feature_size=args.feature_answer_size - 2,
                                                hist_num=args.hist_neighbor_num)
                #    assign_lr()
                print("epoch:", epoch)
                # self.assign_lr(self.sess,self.args.lr * self.args.lr_decay ** epoch)
                overall_loss = 0
                train_generator.shuffle()
                preds, binary_preds, targets = list(), list(), list()
                train_step = 0
                while not train_generator.end:
                    train_step += 1

                    [features_answer_index,target_answers,seq_lens,hist_neighbor_index] = train_generator.next_batch()
                    binary_pred, pred, loss = model.train(sess,features_answer_index,target_answers,seq_lens,hist_neighbor_index)

                    overall_loss += loss
                    for seq_idx, seq_len in enumerate(seq_lens):
                        preds.append(pred[seq_idx, 0:seq_len])
                        binary_preds.append(binary_pred[seq_idx, 0:seq_len])
                        targets.append(target_answers[seq_idx, 0:seq_len])
                # print("\r idx:{0}, overall_loss:{1}".format(train_generator.pos, overall_loss)),
                train_loss = overall_loss / train_step
                preds = np.concatenate(preds)
                binary_preds = np.concatenate(binary_preds)
                targets = np.concatenate(targets)
                auc_value = roc_auc_score(targets, preds)
                accuracy = accuracy_score(targets, binary_preds)
                precision, recall, f_score, _ = precision_recall_fscore_support(targets, binary_preds)
                print("\ntrain loss = {0},auc={1}, accuracy={2}".format(train_loss, auc_value, accuracy))
                write_log(args,model_dir,auc_value, accuracy, epoch, name='train_')

                # if epoch == self.args.num_epochs-1:
                #     self.save(epoch)

                # valid
                valid_generator.reset()
                preds, binary_preds, targets = list(), list(), list()
                valid_step = 0
                #overall_loss = 0
                while not valid_generator.end:
                    valid_step += 1
                    [features_answer_index,target_answers,seq_lens,hist_neighbor_index] = valid_generator.next_batch()
                    binary_pred, pred = model.evaluate(sess,features_answer_index,target_answers,seq_lens,hist_neighbor_index,valid_step)
                    #overall_loss += loss
                    for seq_idx, seq_len in enumerate(seq_lens):
                        preds.append(pred[seq_idx, 0:seq_len])
                        binary_preds.append(binary_pred[seq_idx, 0:seq_len])
                        targets.append(target_answers[seq_idx, 0:seq_len])
                # compute metrics
                #valid_loss = overall_loss / valid_step
                preds = np.concatenate(preds)
                binary_preds = np.concatenate(binary_preds)
                targets = np.concatenate(targets)
                auc_value = roc_auc_score(targets, preds)
                accuracy = accuracy_score(targets, binary_preds)
                precision, recall, f_score, _ = precision_recall_fscore_support(targets, binary_preds)
                print("\nvalid auc={0}, accuracy={1}, precision={2}, recall={3}".format(auc_value, accuracy, precision,
                                                                                        recall))


                write_log(args,model_dir,auc_value, accuracy, epoch, name='valid_')

                if auc_value > best_valid_auc:
                    print('%3.4f to %3.4f' % (best_valid_auc, auc_value))
                    best_valid_auc = auc_value
                    best_epoch = epoch
                    #np.save('feature_embedding.npy', feature_embedding)
                    checkpoint_dir = os.path.join(args.checkpoint_dir, model_dir)
                    save(best_epoch,sess,checkpoint_dir,saver)
#                print(model_dir)
                print(model_dir+"\t"+str(best_valid_auc))




        else:
            if self.load():
                print('CKPT loaded')
            else:
                raise Exception('CKPT need')
            test_data_generator = DataGenerator(args.test_seqs, args.max_step, batch_size=args.batch_size,
                                                feature_size=args.feature_answer_size - 2,
                                                hist_num=args.hist_neighbor_num)
            data_generator.reset()

            correct_times = np.zeros(self.num_skills + 1)
            preds, binary_preds, targets = list(), list(), list()
            while not test_data_generator.end:

                [features_answer_index, target_answers, seq_lens, hist_neighbor_index] = valid_generator.next_batch()
                binary_pred, pred = model.evaluate(sess, features_answer_index, target_answers, seq_lens,
                                                   hist_neighbor_index)
                # overall_loss += loss
                for seq_idx, seq_len in enumerate(seq_lens):
                    preds.append(pred[seq_idx, 0:seq_len])
                    binary_preds.append(binary_pred[seq_idx, 0:seq_len])
                    targets.append(target_answers[seq_idx, 0:seq_len])



            preds = np.concatenate(preds)
            binary_preds = np.concatenate(binary_preds)
            targets = np.concatenate(targets)
            auc_value = roc_auc_score(targets, preds)
            accuracy = accuracy_score(targets, binary_preds)
            precision, recall, f_score, _ = precision_recall_fscore_support(targets, binary_preds)
            print("\ntest auc={0}, accuracy={1}, precision={2}, recall={3}".format(auc_value, accuracy, precision,
                                                                                   recall))
            print(model_dir)
            write_log(args, model_dir, auc_value, accuracy, epoch, name='test_')







def save(global_step,sess,checkpoint_dir,saver):
    model_name = 'GIKT'

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=global_step)
    print('Save checkpoint at %d' % (global_step))


def save_model_dir(args):
    return '{}_{}_{}lr_{}hop_{}sn_{}qn_{}hn_{}nn_{}_{}bound_{}keep_{}'.format(args.dataset,
                                                args.model,args.lr,args.n_hop,args.skill_neighbor_num,args.question_neighbor_num,args.hist_neighbor_num,\
                                                                     args.next_neighbor_num,args.sim_emb,args.att_bound,args.dropout_keep_probs,args.tag)



def write_log(args,model_dir,auc, accuracy, epoch, name='train_'):
    log_path = os.path.join(args.log_dir, name+model_dir+'.csv')
    if not os.path.exists(log_path):
        log_file = open(log_path, 'w')
        log_file.write('Epoch\tAuc\tAccuracy\n')
    else:
        log_file = open(log_path, 'a')

    log_file.write(str(epoch) + '\t' + str(auc) + '\t' + str(accuracy)  + '\n')
    log_file.flush()
