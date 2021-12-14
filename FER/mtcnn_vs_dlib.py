                # draw bounding box and show emotion text on original image
               # cv2.rectangle(image, (face.left() - 5, face.top() - 5), (face.right() + 5, face.bottom() + 5),
               #               (0, 186, 255), 3)
                cv2.rectangle(image, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5),
                              (0, 186, 255), 3)

                #cv2.putText(image, emotion, (int(face.left()) + 10, int(face.top()) - 10), cv2.FONT_HERSHEY_COMPLEX,
                #            0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, emotion, (int(x1) + 10, int(y1) - 10), cv2.FONT_HERSHEY_COMPLEX,
                            0.8, (255, 255, 255), 2, cv2.LINE_AA)
